# alignprop_trainer_fluxfill.py (or updated alignprop_trainer.py)

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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import PyTorchModelHubMixin
from transformers import is_wandb_available

# Import the Flux DDPO pipeline wrapper
from trl.models.modeling_fluxfill_base import DDPOFluxFillPipeline

from .alignprop_config import AlignPropConfig 
from .utils import generate_model_card, get_comet_experiment_url


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


class AlignPropTrainerFlux(PyTorchModelHubMixin): 
    """
    The AlignPropTrainer adapted for Flux-based Diffusion Models using DDPO.
    Optimises diffusion models based on a reward function and generated prompts/inputs.

    Attributes:
        config (`AlignPropConfig`): # TODO: Consider a FluxAlignPropConfig if needed
            Configuration object for AlignPropTrainer.
        reward_function (`Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor]`):
            Reward function evaluating generated images based on prompts and metadata.
            Input signature: reward_function(generated_images, prompts, prompt_metadata) -> rewards
        input_function (`Callable[[], Tuple[str, Optional[str], Any, Any, Dict]]]`):
            Function to generate inputs for the FluxFill pipeline.
            Should return a tuple: (prompt, prompt_2, image, mask_image, metadata)
            `image` and `mask_image` should be processable by the pipeline's image_processor/mask_processor.
        flux_pipeline (`DDPOFluxFillPipeline`):
            FluxFill pipeline wrapper (DDPO-compatible) to be trained.
        image_samples_hook (`Optional[Callable[[Dict[str, Any], int, Any], Any]]`):
            Hook called during training to log image samples.
            Input signature: image_samples_hook(log_dict, global_step, accelerator_tracker)
            log_dict contains keys like 'images' (generated), 'input_images', 'input_masks', 'prompts', 'rewards'.
    """

    _tag_names = ["trl", "alignprop", "flux"] 

    def __init__(
        self,
        config: AlignPropConfig, 
        reward_function: Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        input_function: Callable[[], Tuple[str, Optional[str], Any, Any, Dict]],
        flux_pipeline: DDPOFluxFillPipeline,
        image_samples_hook: Optional[Callable[[Dict[str, Any], int, Any], Any]] = None,
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.input_fn = input_function
        self.reward_fn = reward_function
        self.config = config
        self.image_samples_callback = image_samples_hook
        self.flux_pipeline = flux_pipeline 

        # --- Accelerator Setup ---
        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # Find the most recent checkpoint
                checkpoints = list(
                    filter(lambda x: "checkpoint_" in x and os.path.isdir(os.path.join(self.config.resume_from, x)), os.listdir(self.config.resume_from))
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from, f"checkpoint_{checkpoint_numbers[-1]}"
                )
                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1
            else:
                 # Extract iteration number from checkpoint directory name
                 try:
                     iteration_num = int(os.path.basename(self.config.resume_from).split("_")[-1])
                     accelerator_project_config.iteration = iteration_num + 1
                 except ValueError:
                     logger.warning(f"Could not parse iteration number from checkpoint name: {self.config.resume_from}")


        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(alignprop_trainer_config=config.to_dict())
                if not is_using_tensorboard
                else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")
        set_seed(self.config.seed, device_specific=True)

        self.flux_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # --- Model & Optimizer Setup ---
        trainable_layers = self.flux_pipeline.get_trainable_layers() 

        # Register hooks before preparing the model with Accelerator
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 if configured
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if hasattr(trainable_layers, 'parameters') else trainable_layers
        )

        # Use autocast context from the DDPO pipeline wrapper
        self.autocast = self.flux_pipeline.autocast or self.accelerator.autocast

        # Prepare model and optimizer with Accelerator
        if hasattr(self.flux_pipeline, "use_lora") and self.flux_pipeline.use_lora:
             # If using LoRA, trainable_layers is likely the transformer model instance with adapters
            prepared_model, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
             # We need the parameters *after* prepare for gradient clipping etc.
            self.trainable_layers_params = list(filter(lambda p: p.requires_grad, prepared_model.parameters()))
        else:
             # If not using LoRA, trainable_layers is the transformer model instance
             # Prepare returns the model and optimizer ready for distributed training
            prepared_model, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers_params = list(prepared_model.parameters()) 

        # --- Resume Logic ---
        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            try:
                 self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
            except ValueError:
                 logger.warning(f"Could not determine epoch number from resume path {config.resume_from}. Starting from epoch 0.")
                 self.first_epoch = 0
        else:
            self.first_epoch = 0

    def compute_rewards(self, samples_dict):
        """Computes rewards for the generated samples."""
        rewards, reward_metadata = self.reward_fn(
            samples_dict["images"], samples_dict["prompts"], samples_dict["prompt_metadata"]
        )

        return rewards

    def step(self, epoch: int, global_step: int):
        """Performs a single training step."""
        info = defaultdict(list)
        log_dict_for_callback = {} # Accumulate info for image logging

        self.flux_pipeline.transformer.train() # Set the trainable part to train mode

        for i in range(self.config.train_gradient_accumulation_steps):
            # Enable gradient calculation and autocasting
            with self.accelerator.accumulate(self.flux_pipeline.transformer), self.autocast(), torch.enable_grad():
                # Generate samples with gradients
                samples = self._generate_samples(
                    batch_size=self.config.train_batch_size,
                    with_grad=True,
                )

                rewards = self.compute_rewards(samples)

                samples["rewards"] = rewards # Add rewards to the dictionary

                # --- Loss Calculation using Rewards --- 
                loss = self.calculate_loss(rewards) 

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Clip gradients only when syncing
                    self.accelerator.clip_grad_norm_(
                        self.trainable_layers_params, 
                        self.config.train_max_grad_norm,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log metrics locally 
                rewards_vis = self.accelerator.gather(rewards).detach().cpu().numpy()
                info["reward_mean"].append(rewards_vis.mean())
                info["reward_std"].append(rewards_vis.std())
                info["loss"].append(loss.item()) 

            # Store last samples from accumulation steps for potential logging
            if i == self.config.train_gradient_accumulation_steps - 1:
                 log_dict_for_callback = {
                     "prompts": samples["prompts"],
                     "prompt_metadata": samples["prompt_metadata"],
                     "images": samples["images"], 
                     "input_images": samples["input_images"], 
                     "input_masks": samples["input_masks"],
                     "rewards": samples["rewards"], 
                 }


        # Post-accumulation step actions
        if self.accelerator.sync_gradients:
            # Aggregate and log metrics across all processes
            info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
            info = self.accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch, "step": global_step}) 
            global_step += 1

            # Log images if configured and callback exists
            if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0:
                 if self.accelerator.is_main_process: # Only log images from the main process
                     # Check if there are any trackers initialized
                     if self.accelerator.trackers:
                         # Detach tensors in the log dictionary before passing to callback
                         log_dict_detached = {}
                         for k, v in log_dict_for_callback.items():
                             if isinstance(v, torch.Tensor):
                                 log_dict_detached[k] = v.detach().cpu()
                             elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                                 log_dict_detached[k] = [t.detach().cpu() for t in v]
                             else:
                                 log_dict_detached[k] = v # Keep non-tensors as is

                         self.image_samples_callback(log_dict_detached, global_step, self.accelerator.trackers[0])
                     else:
                         logger.warning("Image logging callback defined, but no tracker initialized (e.g., --log_with wandb). Skipping image logging.")

        # Save checkpoint
        if epoch != 0 and epoch % self.config.save_freq == 0: 
            # Ensure all processes are synchronized before the main process attempts to save.
            self.accelerator.wait_for_everyone() 
            
            if self.accelerator.is_main_process:
                logger.info(f"Saving checkpoint for step {global_step}")
                self.accelerator.save_state() # Only main process saves

        return global_step

    def calculate_loss(self, rewards):
        """
        Calculate the AlignProp loss. Loss = 10.0 - mean(rewards).
        Assumes higher reward is better.
        """
        loss = 10.0 - rewards.mean() 
        return loss

    def _setup_optimizer(self, trainable_layers_parameters):
        """Sets up the optimizer based on the configuration."""
        if self.config.train_use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("bitsandbytes not installed. Please install it with `pip install bitsandbytes`")
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        else:
            optimizer_cls = torch.optim.AdamW
            logger.info("Using standard AdamW optimizer.")

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    # --- Accelerator Hooks ---
    def _save_model_hook(self, models, weights, output_dir):
        """Hook called by Accelerator before saving state."""
        logger.debug(f"Saving model checkpoint to {output_dir}")

        self.flux_pipeline.save_checkpoint(models, weights, output_dir)


    def _load_model_hook(self, models, input_dir):
        """Hook called by Accelerator before loading state."""
        logger.debug(f"Loading model checkpoint from {input_dir}")

        self.flux_pipeline.load_checkpoint(models, input_dir)


    def _generate_samples(self, batch_size, with_grad=True):
        """Generates samples using the FluxFill pipeline."""
        samples_dict = {}

        # Get inputs from the input function
        # Returns list of tuples: [(prompt, prompt2, img, mask, meta), ...]
        input_data = [self.input_fn() for _ in range(batch_size)]
        prompts, prompts_2, images_in, masks_in, metadata = zip(*input_data)

        # Ensure prompts and prompts_2 are lists
        prompts = list(prompts)
        prompts_2 = list(prompts_2) # Can contain None values if not provided

        # Select the appropriate generation function
        generate_fn = self.flux_pipeline.rgb_with_grad if with_grad else self.flux_pipeline.__call__

        # Determine the prompt format based on batch size
        final_prompt = prompts[0] if batch_size == 1 else prompts
        final_prompt_2 = prompts_2[0] if batch_size == 1 and prompts_2 else prompts_2

        # Call the DDPO pipeline's generation function
        # It handles internal encoding and the diffusion loop
        output = generate_fn(
            prompt=final_prompt,
            prompt_2=final_prompt_2,
            image=list(images_in), 
            mask_image=list(masks_in), 
            height=self.config.image_height, 
            width=self.config.image_width,   
            num_inference_steps=self.config.sample_num_steps,
            guidance_scale=self.config.sample_guidance_scale,
            max_sequence_length=512, 
            # --- Grad specific args (passed only if with_grad=True) ---
            #truncated_backprop=self.config.truncated_backprop if with_grad else False,
            truncated_backprop_rand=self.config.truncated_backprop_rand if with_grad else False,
            truncated_backprop_timestep=self.config.truncated_backprop_timestep if with_grad else 0,
            truncated_rand_backprop_minmax=self.config.truncated_rand_backprop_minmax if with_grad else (0,0),
            gradient_checkpoint=False if with_grad else False,
            # --- Output format ---
            output_type="pt",
        )

        # Store results
        samples_dict["prompts"] = prompts
        samples_dict["prompts_2"] = prompts_2
        samples_dict["prompt_metadata"] = list(metadata)
        samples_dict["images"] = output.images 
        samples_dict["input_images"] = list(images_in) 
        samples_dict["input_masks"] = list(masks_in)  

        return samples_dict

    def train(self, epochs: Optional[int] = None):
        """Main training loop."""
        total_epochs = epochs if epochs is not None else self.config.num_epochs
        logger.info(f"Starting training for {total_epochs} epochs.")
        global_step = 0 

         # Load initial state if resuming
        if self.config.resume_from:
            try:
                global_step = self.accelerator.project_configuration.iteration or 0
                logger.info(f"Resumed training. Starting from epoch {self.first_epoch}, global_step {global_step}")
            except Exception as e:
                logger.warning(f"Could not retrieve global_step from Accelerator state: {e}. Starting step 0.")


        for epoch in range(self.first_epoch, total_epochs):
            logger.info(f"Starting epoch {epoch}")
            global_step = self.step(epoch, global_step)
            logger.info(f"Finished epoch {epoch}. Global step: {global_step}")

        logger.info("Training finished.")

        if self.accelerator.is_main_process:
            logger.info("Saving final model state.")
            self.accelerator.save_state() # Save final state


    def _save_pretrained(self, save_directory):
        """Save the trained pipeline."""
        logger.info(f"Saving final pipeline to {save_directory}")
        self.flux_pipeline.save_pretrained(save_directory)
        # Optionally create a model card
        # self.create_model_card(save_directory) 

    # --- TODO Model Card Generation (Adapt as needed) ---
    def create_model_card(
        self,
        output_dir: str, 
        model_name: Optional[str] = None,
        base_model_name: Optional[str] = None, 
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """Creates a draft model card."""
        if not self.accelerator.is_main_process:
            return

        # Try to get base model info from the pipeline if available
        if base_model_name is None and hasattr(self.flux_pipeline, 'pretrained_model'):
             base_model = self.flux_pipeline.pretrained_model
        else:
             base_model = base_model_name or "Unknown Flux Model"

        hub_model_id = f"{self.config.tracker_project_name}/{model_name}" if model_name else None # Construct tentative hub id

        final_tags = ["flux", "fluxfill", "alignprop", "trl"] # Start with base tags
        if isinstance(tags, str):
            final_tags.append(tags)
        elif isinstance(tags, list):
            final_tags.extend(tags)
        # Add unique tags
        final_tags = sorted(list(set(final_tags)))

        citation = textwrap.dedent("""\
        @article{prabhudesai2024aligning,
            title        = {{Aligning Text-to-Image Diffusion Models with Reward Backpropagation}},
            author       = {Mihir Prabhudesai and Anirudh Goyal and Deepak Pathak and Katerina Fragkiadaki},
            year         = 2024,
            eprint       = {arXiv:2310.03739}
        }""")

        try: 
            wandb_url = wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None
        except Exception:
            wandb_url = None

        try: 
            comet_url = get_comet_experiment_url()
        except Exception:
            comet_url = None

        # Use the utility function to generate the card content
        model_card_content = generate_model_card(
            model_name=model_name,
            base_model=base_model,
            model_description="A Flux-Fill model fine-tuned using AlignProp.",
            hub_model_id=hub_model_id,
            license="apache-2.0", 
            dataset_name=dataset_name,
            tags=final_tags,
            wandb_url=wandb_url,
            comet_url=comet_url,
            trainer_name="AlignPropTrainerFlux",
            trainer_citation=citation,
            paper_title="Aligning Text-to-Image Diffusion Models with Reward Backpropagation",
            paper_id="2310.03739",
        )

        # Save the model card
        card_path = os.path.join(output_dir, "README.md")
        try:
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)
            logger.info(f"Model card saved to {card_path}")
        except Exception as e:
            logger.error(f"Failed to save model card: {e}")