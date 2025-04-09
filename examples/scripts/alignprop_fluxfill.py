# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Total Batch size = 128 = 4 (num_gpus) * 8 (per_device_batch) * 4 (accumulation steps)
Feel free to reduce batch size or increasing truncated_rand_backprop_min to a higher value to reduce memory usage.

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/scripts/alignprop.py \
    --num_epochs=20 \
    --train_gradient_accumulation_steps=4 \
    --sample_num_steps=50 \
    --train_batch_size=8 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"

"""

from dataclasses import dataclass, field

import numpy as np
from transformers import HfArgumentParser, CLIPProcessor, CLIPModel
from trl import AlignPropConfig, AlignPropTrainerFlux, DefaultDDPOFluxFillPipeline
from trl.models.auxiliary_modules import aesthetic_scorer
from PIL import Image
import random
import torch
import torch.nn.functional as F
import accelerate

logger = accelerate.logging.get_logger(__name__)

@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"alignprop-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
        clip_model_id (`str`, *optional*, defaults to `"openai/clip-vit-large-patch14"`):
            CLIP model ID for similarity reward.
        clip_reward_scale (`float`, *optional*, defaults to `1.0`):
            Scale factor for the CLIP similarity reward component.
    """

    pretrained_model: str = field(
        default="black-forest-labs/FLUX.1-Fill-dev", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="alignprop-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    clip_model_id: str = field(default="openai/clip-vit-large-patch14", metadata={"help": "CLIP model ID for similarity reward."})
    clip_reward_scale: float = field(default=20.0, metadata={"help": "Scale factor for the CLIP similarity reward component."})
    aesthetic_reward_scale: float = field(default=0.4, metadata={"help": "Scale factor for the aesthetic reward component."})


animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    """Generate prompts for training."""

    # Create white images for testing
    image = Image.new('RGB', (512, 512), color='white')
    mask_image = Image.new('RGB', (512, 512), color='white')
    
    # Return the prompts and images
    return random.choice(animals), None, image, mask_image, {} 

def image_outputs_logger(image_pair_data, global_step, accelerate_logger):

    result = {}
    images, prompts, _ = [image_pair_data["images"], image_pair_data["prompts"], image_pair_data["rewards"]]
    for i, image in enumerate(images[:4]):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )


def combined_reward_function(
    images: torch.Tensor, 
    prompts: tuple[str], 
    metadata: tuple[dict],
    aesthetic_model: torch.nn.Module,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    clip_scale: float = 1.0,
    aesthetic_scale: float = 1.0,
):
    """
    Calculates a combined reward: aesthetic_score + scale * clip_similarity.
    Ensures differentiability through the aesthetic scorer and CLIP image encoder.
    Moves models to correct device at runtime.
    """
    # Determine target device from input images
    target_device = images.device 
    target_dtype = images.dtype 

    # 1. Aesthetic Score

    try:
        aesthetic_rewards, metadata = aesthetic_model(images,prompts,metadata) 
    except Exception as e:
         print(f"Error getting aesthetic score: {e}")

         return torch.zeros(images.shape[0], device=target_device, dtype=target_dtype)

    # 2. CLIP Similarity
    clip_model.to(target_device)
    try:
        # Preprocess images (needs grads)
        image_inputs = clip_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs.pixel_values.to(target_device, dtype=clip_model.dtype) 

        # Preprocess text (no grads needed)
        with torch.no_grad():
            text_inputs = clip_processor(
                text=list(prompts), padding=True, truncation=True, return_tensors="pt"
            ).to(target_device)
            text_embeds = clip_model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Get Image Embeddings (needs grads)
        image_embeds = clip_model.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Calculate Cosine Similarity 
        clip_similarity = F.cosine_similarity(image_embeds, text_embeds.to(image_embeds.dtype))

    except Exception as e:
        print(f"Error during CLIP processing: {e}")
        clip_similarity = torch.zeros_like(aesthetic_rewards)

    # 3. Combine Rewards 
    clip_scale_tensor = torch.tensor(clip_scale, device=target_device, dtype=aesthetic_rewards.dtype)
    combined_rewards = aesthetic_rewards*aesthetic_scale + clip_scale_tensor * clip_similarity

    print(f"[REWARD DBG] Aesthetic: {aesthetic_rewards.mean().item():.3f} | CLIP Sim: {clip_similarity.mean().item():.3f} | Combined: {combined_rewards.mean().item():.3f}")

    return combined_rewards, metadata


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save",
    }

    # --- Load Models for Reward ---
    # 1. Aesthetic Scorer 
    print("Loading Aesthetic Scorer...")
    aesthetic_model_instance = aesthetic_scorer( # Call the factory function
        script_args.hf_hub_aesthetic_model_id, 
        script_args.hf_hub_aesthetic_model_filename
    )
    print("Aesthetic Scorer loaded.")
    # 2. CLIP Model & Processor
    print(f"Loading CLIP model: {script_args.clip_model_id}")
    clip_processor = CLIPProcessor.from_pretrained(script_args.clip_model_id)
    clip_model = CLIPModel.from_pretrained(script_args.clip_model_id)
    # Freeze CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad_(False)
    print("CLIP model loaded and frozen.")
    # -----------------------------

    pipeline = DefaultDDPOFluxFillPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    pipeline.to("cuda")

    # --- Create the final reward function with models baked in ---
    reward_fn = lambda images, prompts, metadata: combined_reward_function(
        images=images,
        prompts=prompts,
        metadata=metadata,
        aesthetic_model=aesthetic_model_instance,
        clip_model=clip_model,
        clip_processor=clip_processor,
        clip_scale=script_args.clip_reward_scale,
        aesthetic_scale=script_args.aesthetic_reward_scale,
    )
    # -------------------------------------------------------------

    trainer = AlignPropTrainerFlux(
        training_args,
        reward_fn,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
