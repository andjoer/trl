# modeling_flux_fill_ddpo.py

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

import contextlib
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from diffusers import FluxFillPipeline, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.pipelines.flux.pipeline_flux_fill import retrieve_latents, retrieve_timesteps 
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.utils import is_peft_available

from diffusers.utils import (
    logging,
    is_torch_xla_available,
) 
from diffusers.utils.torch_utils import randn_tensor 


if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

# --- Data Classes ---
@dataclass
class DDPOFluxFillPipelineOutput:
    """
    Output class for the FluxFill pipeline modified for DDPO.

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`list[torch.Tensor]`):
            The list of latents computed at each step of the denoising process. The first element is the initial latent.
        log_probs (`list[torch.Tensor]`):
            The log probabilities computed at each step of the denoising process.
    """

    images: torch.Tensor
    latents: List[torch.Tensor]
    log_probs: List[torch.Tensor]


@dataclass
class DDPOFluxFillSchedulerOutput:
    """
    Output class for the FluxFill scheduler modified for DDPO.

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape depends on whether latents are packed.
        log_probs (`torch.Tensor`):
            Log probability of the above-mentioned sample. Shape: `(batch_size)`.
            NOTE: For FlowMatchEulerDiscreteScheduler, this is currently a placeholder (zeros).
    """

    latents: torch.Tensor
    log_probs: torch.Tensor

# --- Abstract Base Class ---
class DDPOFluxFillPipeline:
    """
    Abstract base class for FluxFill pipelines compatible with DDPOTrainer.
    """

    def __call__(self, *args, **kwargs) -> DDPOFluxFillPipelineOutput:
        raise NotImplementedError

    def rgb_with_grad(self, *args, **kwargs) -> DDPOFluxFillPipelineOutput:
         raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOFluxFillSchedulerOutput:
        raise NotImplementedError

    @property
    def transformer(self):
        raise NotImplementedError

    @property
    def vae(self):
        raise NotImplementedError

    @property
    def tokenizer(self):
        raise NotImplementedError

    @property
    def tokenizer_2(self):
         raise NotImplementedError

    @property
    def scheduler(self):
        raise NotImplementedError

    @property
    def text_encoder(self):
        raise NotImplementedError

    @property
    def text_encoder_2(self):
        raise NotImplementedError

    @property
    def autocast(self):
        raise NotImplementedError

    def set_progress_bar_config(self, *args, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError

    def get_trainable_layers(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

# --- Scheduler Step Modification ---
def scheduler_step(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    **kwargs,
) -> DDPOFluxFillSchedulerOutput:
    """
    Applies one step of the denoising process. modification for DDPO compatibility.

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from the learned diffusion model (most often the predicted velocity).
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        **kwargs:
             Other arguments passed to the original scheduler step.

    Returns:
        `DDPOFluxFillSchedulerOutput`: the predicted sample at the previous timestep and the log probability of the sample
    """
    # Call the original scheduler step
    prev_sample_tuple = self.step(model_output, timestep, sample, return_dict=False, **kwargs)
    prev_sample = prev_sample_tuple[0]

    # ---- Log Probability Calculation ----
    # FIXME: This is a placeholder! Calculating accurate log probs for FlowMatchEulerDiscreteScheduler
    # is non-trivial as it's an ODE solver step, not a simple Gaussian transition like DDIM.
    # Returning zeros as a placeholder. This might be sufficient for some DDPO tasks.
    # A more accurate calculation would require analyzing the underlying ODE/SDE or Flow Matching objective.
    # alignprop does not use log probs
    log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
    # ------------------------------------

    return DDPOFluxFillSchedulerOutput(latents=prev_sample, log_probs=log_prob)

# --- Pipeline Step Modifications ---

@torch.no_grad()
def pipeline_step(
    self: FluxFillPipeline, 
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    image: Optional[torch.FloatTensor] = None,
    mask_image: Optional[torch.FloatTensor] = None,
    masked_image_latents: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 30.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True, # Keep for consistency, but DDPO output is fixed
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
) -> DDPOFluxFillPipelineOutput:
    r"""
    Function invoked when calling the pipeline for generation, modified for DDPO.
    Returns images, latents at each step, and log probabilities at each step.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
        prompt_2 (`str` or `List[str]`, *optional*):
             The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used.
        image (`torch.Tensor`...): See FluxFillPipeline.__call__
        mask_image (`torch.Tensor`...): See FluxFillPipeline.__call__
        masked_image_latents (`torch.Tensor`...): See FluxFillPipeline.__call__
        height (`int`, *optional*): See FluxFillPipeline.__call__
        width (`int`, *optional*): See FluxFillPipeline.__call__
        strength (`float`, *optional*, defaults to 1.0): See FluxFillPipeline.__call__
        num_inference_steps (`int`, *optional*, defaults to 50): See FluxFillPipeline.__call__
        sigmas (`List[float]`, *optional*): See FluxFillPipeline.__call__
        guidance_scale (`float`, *optional*, defaults to 30.0): See FluxFillPipeline.__call__
        num_images_per_prompt (`int`, *optional*, defaults to 1): See FluxFillPipeline.__call__
        generator (`torch.Generator`...): See FluxFillPipeline.__call__
        latents (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        prompt_embeds (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        output_type (`str`, *optional*, defaults to `"pil"`): See FluxFillPipeline.__call__
        return_dict (`bool`, *optional*, defaults to `True`): Ignored, always returns DDPOFluxFillPipelineOutput.
        joint_attention_kwargs (`dict`, *optional*): See FluxFillPipeline.__call__
        callback_on_step_end (`Callable`, *optional*): See FluxFillPipeline.__call__
        callback_on_step_end_tensor_inputs (`List`, *optional*): See FluxFillPipeline.__call__
        max_sequence_length (`int` defaults to 512): See FluxFillPipeline.__call__

    Returns:
        `DDPOFluxFillPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt=prompt,
        prompt_2=prompt_2,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
        image=image,
        mask_image=mask_image,
        masked_image_latents=masked_image_latents
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    init_image = self.image_processor.preprocess(image, height=height, width=width)
    init_image = init_image.to(dtype=torch.float32)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    dtype = self.transformer.dtype 

    # 3. Prepare prompt embeddings
    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2, prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds, device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length, lora_scale=lora_scale,
    )

    # 4. Prepare timesteps
    # Use default sigmas if not provided, then calculate shift and retrieve timesteps
    if sigmas is None:
         # Default sigmas based on num_inference_steps 
        sigmas = np.linspace(self.scheduler.config.sigma_max, self.scheduler.config.sigma_min, num_inference_steps)

    image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps_cat = retrieve_timesteps( 
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    # Adjust timesteps based on strength
    timesteps_adjusted, num_inference_steps_adjusted = self.retrieve_timesteps(num_inference_steps, strength, device)

    timesteps = timesteps_adjusted
    num_inference_steps = num_inference_steps_adjusted
    # --- End inline get_timesteps logic ---


    if num_inference_steps < 1:
        raise ValueError(f"Invalid num_inference_steps {num_inference_steps} after strength adjustment.")

    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)


    # 5. Prepare latent variables
    num_channels_latents = self.vae.config.latent_channels
    # Call with arguments matching FluxPipeline.prepare_latents definition

    base_device = self._execution_device 
    latents, latent_image_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=base_device,
        generator=generator,
        latents=latents,
    )

    # 6. Prepare mask and masked image latents
    if masked_image_latents is not None:

        masked_image_latents = masked_image_latents.to(base_device)
    else:
        # Preprocess images and masks
        mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)
        masked_image = init_image * (1 - mask_image)
        masked_image = masked_image.to(device=base_device, dtype=prompt_embeds.dtype)

        height, width = init_image.shape[-2:]

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_image,
            masked_image,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            base_device, 
            generator,
        )
       
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

   
    transformer_device = self.transformer.device

    # Concatenate latents and masked_image_latents for transformer input
    transformer_input_hidden_states = torch.cat((latents, masked_image_latents), dim=2)

    transformer_input_hidden_states = transformer_input_hidden_states.to(transformer_device)
    latent_image_ids = latent_image_ids.to(transformer_device)
    text_ids = text_ids.to(transformer_device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_device)
    prompt_embeds = prompt_embeds.to(transformer_device)


    # --- Classifier-Free Guidance Setup ---
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=transformer_device, dtype=torch.float32) # Use transformer_device
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
    else:
        guidance = None

    # --- Timestep setup  ---
    timestep = timestep.expand(latents.shape[0]).to(transformer_device) # Use transformer_device

    # --- Grad Enabled Transformer Step ---
    with torch.enable_grad():
        noise_pred = self.transformer(
            hidden_states=transformer_input_hidden_states, 
            timestep=timestep / 1000,  
            guidance=guidance, 
            pooled_projections=pooled_prompt_embeds, 
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, 
            img_ids=latent_image_ids, 
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

    # 7. Denoising loop
    all_latents = [latents] # Store initial latents
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension
            timestep_expanded = t.expand(latents.shape[0]).to(dtype)

            # Model prediction
            # Concatenate input latents and mask+masked_image along the channel dim (dim=2 for packed)
            model_input = torch.cat((latents, masked_image_latents), dim=2)

            noise_pred = self.transformer(
                hidden_states=model_input,
                timestep=timestep_expanded / 1000, # Flux expects scaled timestep
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # Scheduler step using the modified function
            scheduler_output = scheduler_step(self.scheduler, noise_pred, t, latents) 
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # Callback handling
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                # pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds) 

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    # 8. Post-process image
    if not output_type == "latent":
        # Unpack latents before decoding
        latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image_decoded = self.vae.decode(latents_unpacked, return_dict=False)[0]
        image = self.image_processor.postprocess(image_decoded, output_type=output_type)
    else:
        image = latents # Return packed latents if output_type is 'latent'

    # Offload models
    self.maybe_free_model_hooks()

    # Concatenate log probs for the whole trajectory
    if all_log_probs:
        all_log_probs_tensor = torch.stack(all_log_probs, dim=1) # (batch_size, num_steps)
    else:
        all_log_probs_tensor = torch.empty((batch_size, 0), device=device, dtype=dtype)


    return DDPOFluxFillPipelineOutput(images=image, latents=all_latents, log_probs=all_log_probs_tensor)


# Function to get RGB image with gradients attached
def pipeline_step_with_grad(
    pipeline: FluxFillPipeline, # Takes the FluxFillPipeline instance
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    image: Optional[torch.FloatTensor] = None,
    mask_image: Optional[torch.FloatTensor] = None,
    masked_image_latents: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 30.0,
    truncated_backprop: bool = True,
    truncated_backprop_rand: bool = True,
    gradient_checkpoint: bool = True,
    truncated_backprop_timestep: int = 49, 
    truncated_rand_backprop_minmax: tuple = (0, 50), 
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True, # Keep for consistency, but DDPO output is fixed
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
) -> DDPOFluxFillPipelineOutput:
    r"""
    Function to get RGB image with gradients attached to the model weights, modified for DDPO.
    Returns images, latents at each step, and log probabilities at each step.

    Args:
        pipeline (`FluxFillPipeline`): The FluxFill pipeline instance.
        prompt (`str` or `List[str]`, *optional*): See FluxFillPipeline.__call__
        prompt_2 (`str` or `List[str]`, *optional*): See FluxFillPipeline.__call__
        image (`torch.Tensor`...): See FluxFillPipeline.__call__
        mask_image (`torch.Tensor`...): See FluxFillPipeline.__call__
        masked_image_latents (`torch.Tensor`...): See FluxFillPipeline.__call__
        height (`int`, *optional*): See FluxFillPipeline.__call__
        width (`int`, *optional*): See FluxFillPipeline.__call__
        strength (`float`, *optional*, defaults to 1.0): See FluxFillPipeline.__call__
        num_inference_steps (`int`, *optional*, defaults to 50): See FluxFillPipeline.__call__
        sigmas (`List[float]`, *optional*): See FluxFillPipeline.__call__
        guidance_scale (`float`, *optional*, defaults to 30.0): See FluxFillPipeline.__call__
        truncated_backprop (`bool`, *optional*, defaults to True): Enable truncated backpropagation.
        truncated_backprop_rand (`bool`, *optional*, defaults to True): Enable randomized truncated backpropagation.
        gradient_checkpoint (`bool`, *optional*, defaults to True): Enable gradient checkpointing for the transformer.
        truncated_backprop_timestep (`int`, *optional*, defaults to 49): Timestep index for fixed truncation.
        truncated_rand_backprop_minmax (`Tuple`, *optional*, defaults to (0,50)): Range for randomized truncation.
        num_images_per_prompt (`int`, *optional*, defaults to 1): See FluxFillPipeline.__call__
        generator (`torch.Generator`...): See FluxFillPipeline.__call__
        latents (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        prompt_embeds (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*): See FluxFillPipeline.__call__
        output_type (`str`, *optional*, defaults to `"pil"`): See FluxFillPipeline.__call__
        return_dict (`bool`, *optional*, defaults to `True`): Ignored, always returns DDPOFluxFillPipelineOutput.
        joint_attention_kwargs (`dict`, *optional*): See FluxFillPipeline.__call__
        callback_on_step_end (`Callable`, *optional*): See FluxFillPipeline.__call__
        callback_on_step_end_tensor_inputs (`List`, *optional*): See FluxFillPipeline.__call__
        max_sequence_length (`int` defaults to 512): See FluxFillPipeline.__call__

    Returns:
        `DDPOFluxFillPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities.
    """
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # --- Pre-computation (no gradients needed here) ---
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            mask_image=mask_image,
            masked_image_latents=masked_image_latents
        )

        print(f"pipeline device: {pipeline.device}")
        pipeline._guidance_scale = guidance_scale
        pipeline._joint_attention_kwargs = joint_attention_kwargs
        pipeline._interrupt = False

        init_image = pipeline.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        dtype = pipeline.transformer.dtype

        # 3. Prepare prompt embeddings
        print(f"prompt: {prompt}")
        lora_scale = (
            pipeline.joint_attention_kwargs.get("scale", None) if pipeline.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length, lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        #if sigmas is None:
            #sigmas = np.linspace(pipeline.scheduler.config.sigma_max, pipeline.scheduler.config.sigma_min, num_inference_steps)

        image_seq_len = (int(height) // pipeline.vae_scale_factor // 2) * (int(width) // pipeline.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps_cat = retrieve_timesteps( # Use FluxFill retrieve_timesteps
            pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        #timesteps, num_inference_steps = pipeline.retrieve_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(f"Invalid num_inference_steps {num_inference_steps} after strength adjustment.")
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = pipeline.vae.config.latent_channels
        # Call with arguments matching FluxPipeline.prepare_latents definition
        latents, latent_image_ids = pipeline.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 6. Prepare mask and masked image latents
        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.to(device, dtype=dtype)
        else:
            mask_image_processed = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)
            mask_image_processed = mask_image_processed.to(device=device, dtype=dtype)

            masked_image = init_image.to(device=device, dtype=dtype) * (1 - mask_image_processed)

            _height, _width = init_image.shape[-2:]
            mask, masked_image_packed = pipeline.prepare_mask_latents(
                mask_image_processed, masked_image, batch_size, num_channels_latents,
                num_images_per_prompt, _height, _width, dtype, device, generator,
            )
            masked_image_latents = torch.cat((masked_image_packed, mask), dim=-1)

    # --- Denoising Loop (with potential gradients) ---
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Handle guidance embedding for Flux
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    all_latents = [latents.detach()] # Store initial latents (detached)
    all_log_probs = []
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            # broadcast to batch dimension
            timestep_expanded = t.expand(latents.shape[0]).to(dtype)

            # Concatenate input latents and mask+masked_image along the channel dim
            model_input = torch.cat((latents, masked_image_latents.detach()), dim=2) # Detach mask input

            # Predict the noise residual (potentially with gradient checkpointing)
            if gradient_checkpoint:
                # We need to wrap the transformer call for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Match the expected signature of transformer.forward
                        return module(
                            hidden_states=inputs[0],
                            timestep=inputs[1],
                            guidance=inputs[2],
                            pooled_projections=inputs[3],
                            encoder_hidden_states=inputs[4],
                            txt_ids=inputs[5],
                            img_ids=inputs[6],
                            joint_attention_kwargs=inputs[7],
                            return_dict=False 
                        )
                    return custom_forward

                noise_pred = checkpoint.checkpoint(
                    create_custom_forward(pipeline.transformer),
                    model_input,
                    timestep_expanded / 1000,
                    guidance,
                    pooled_prompt_embeds.detach(), # Detach embeds if not training text encoders
                    prompt_embeds.detach(),        # Detach embeds if not training text encoders
                    text_ids.detach(),             # Detach ids
                    latent_image_ids.detach(),     # Detach ids
                    joint_attention_kwargs,
                    use_reentrant=False, # Recommended for PyTorch >= 1.11
                )[0]
            else:
                 noise_pred = pipeline.transformer(
                    hidden_states=model_input,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds.detach(), # Detach embeds
                    encoder_hidden_states=prompt_embeds.detach(),   # Detach embeds
                    txt_ids=text_ids.detach(),
                    img_ids=latent_image_ids.detach(),
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                 )[0]


            # Truncated Backpropagation Logic
            if truncated_backprop:
                if truncated_backprop_rand:
                    rand_timestep = random.randint(
                        truncated_rand_backprop_minmax[0], truncated_rand_backprop_minmax[1]
                    )
                    if i < rand_timestep:
                        noise_pred = noise_pred.detach()
                else:
                    if i < truncated_backprop_timestep:
                        noise_pred = noise_pred.detach()

            # Scheduler step
            scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents)
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs.detach()

            if i == len(timesteps) - 1:
                 print(f"[DEBUG Step {i}] latents after scheduler requires_grad: {latents.requires_grad}, grad_fn: {latents.grad_fn is not None}") # DBG 2 (Last Step)

            all_latents.append(latents.detach()) # Store detached latents
            all_log_probs.append(log_prob)

            # Callback handling
            if callback_on_step_end is not None:
                 with torch.no_grad():
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        local_val = locals()[k]
                        callback_kwargs[k] = local_val.detach() if isinstance(local_val, torch.Tensor) else local_val

                    callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)

                    # Handle potential updates from callback 
                    latents_callback = callback_outputs.pop("latents", latents.detach())
                    if not torch.equal(latents_callback, latents.detach()):
                         logger.warning("Callback modified latents, but gradients won't flow back through this modification.")

                    prompt_embeds_callback = callback_outputs.pop("prompt_embeds", prompt_embeds.detach())
                    if not torch.equal(prompt_embeds_callback, prompt_embeds.detach()):
                         logger.warning("Callback modified prompt_embeds.")
                         prompt_embeds = prompt_embeds_callback 
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()


    # --- AFTER LOOP --- 

    # --- Post-processing --- 
    if not output_type == "latent":
        latents_unpacked = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        
        latents_unpacked = (latents_unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        image_decoded = pipeline.vae.decode(latents_unpacked, return_dict=False)[0]
        
        image_output = pipeline.image_processor.postprocess(image_decoded, output_type=output_type)

    else:
        image_output = latents 

    # --- Final Cleanup  --- 
    with torch.no_grad():
        # Offload models
        pipeline.maybe_free_model_hooks()

        # Concatenate log probs
        if all_log_probs:
            all_log_probs_tensor = torch.stack(all_log_probs, dim=1)
        else:
            all_log_probs_tensor = torch.empty((batch_size, 0), device=device, dtype=dtype)

    return DDPOFluxFillPipelineOutput(images=image_output, latents=all_latents, log_probs=all_log_probs_tensor)


# --- Default Wrapper Class ---
class DefaultDDPOFluxFillPipeline(DDPOFluxFillPipeline):
    """
    Default implementation of the DDPOFluxFillPipeline based on Diffusers' FluxFillPipeline.
    """
    def __init__(self, pretrained_model_name: str, *, pretrained_model_revision: str = "main", use_lora: bool = True):

        try:
            self.flux_pipeline = FluxFillPipeline.from_pretrained(
                pretrained_model_name, revision=pretrained_model_revision
            )
        except Exception as e:
            raise ValueError(
                f"Could not load pretrained FluxFillPipeline from {pretrained_model_name} with revision {pretrained_model_revision}. Error: {e}"
            )

        self.pretrained_model = pretrained_model_name
        self.pretrained_revision = pretrained_model_revision
        self.use_lora = use_lora

        # Attempt to load LoRA weights specifically for the transformer
        if use_lora:
            try:
                 # FluxLoraLoaderMixin methods handle transformer LoRA
                self.flux_pipeline.load_lora_weights(
                    pretrained_model_name,
                    revision=pretrained_model_revision,
                    weight_name="pytorch_lora_weights.safetensors"
                )
                logger.info(f"Successfully loaded LoRA weights for transformer from {pretrained_model_name}")
            except OSError:
                 warnings.warn(
                    f"LoRA weights (`pytorch_lora_weights.safetensors` or similar) not found for transformer in {pretrained_model_name}. "
                    "If you intended to use LoRA, check the path and revision. Initializing new LoRA layers.",
                    UserWarning,
                 )
                 # We'll add adapters in get_trainable_layers if they weren't loaded
            except Exception as e:
                 warnings.warn(f"Failed to load LoRA weights: {e}. Initializing new LoRA layers.", UserWarning)

        # Ensure the correct scheduler type (already done in FluxFillPipeline init)
        if not isinstance(self.flux_pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
             warnings.warn(
                f"Expected FlowMatchEulerDiscreteScheduler, but found {type(self.flux_pipeline.scheduler)}. "
                "Log probability calculation might be inaccurate.", UserWarning
             )

        # Memory optimization: Keep VAE and Text Encoders frozen by default
        self.flux_pipeline.vae.requires_grad_(False)
        self.flux_pipeline.text_encoder.requires_grad_(False)
        self.flux_pipeline.text_encoder_2.requires_grad_(False)
        # Only make transformer trainable if not using LoRA for it
        self.flux_pipeline.transformer.requires_grad_(not self.use_lora)

    def to(self, device):
        """Moves the underlying pipeline and components to the specified device, applying MPS workarounds if needed."""
        logger.info(f"Moving underlying FluxFillPipeline to device: {device}")
        self.flux_pipeline.to(device)


        self.flux_pipeline.transformer.to(device)
        self.flux_pipeline.vae.to(device)
             # Ensure they exist before moving
        if hasattr(self.flux_pipeline, 'text_encoder') and self.flux_pipeline.text_encoder is not None:
            self.flux_pipeline.text_encoder.to(device)
        if hasattr(self.flux_pipeline, 'text_encoder_2') and self.flux_pipeline.text_encoder_2 is not None:
            self.flux_pipeline.text_encoder_2.to(device)
        # --------------------------------------------------
        return self # Allow chaining

    def __call__(self, *args, **kwargs) -> DDPOFluxFillPipelineOutput:
        """Runs the generation process without gradients."""
        return pipeline_step(self.flux_pipeline, *args, **kwargs)

    def rgb_with_grad(self, *args, **kwargs) -> DDPOFluxFillPipelineOutput:
         """Runs the generation process with gradients enabled."""
         return pipeline_step_with_grad(self.flux_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs) -> DDPOFluxFillSchedulerOutput:
        """Performs one scheduler step and calculates log probability."""
        return scheduler_step(self.flux_pipeline.scheduler, *args, **kwargs)

    # --- Properties to access components ---
    @property
    def transformer(self):
        return self.flux_pipeline.transformer

    @property
    def vae(self):
        return self.flux_pipeline.vae

    @property
    def tokenizer(self):
        return self.flux_pipeline.tokenizer

    @property
    def tokenizer_2(self):
         return self.flux_pipeline.tokenizer_2

    @property
    def scheduler(self):
        return self.flux_pipeline.scheduler

    @property
    def text_encoder(self):
        return self.flux_pipeline.text_encoder

    @property
    def text_encoder_2(self):
        return self.flux_pipeline.text_encoder_2

    @property
    def autocast(self):
        # Enable autocast if transformer is in half precision and not using LoRA (or if LoRA handles casting)
        # LoRA layers are often kept in float32, so autocast might not be needed if only LoRA is trained.
        # Adjust this logic based on specific training setup (e.g., mixed precision used).
        # Safest default might be nullcontext if using LoRA, or if unsure.
        # return torch.cuda.amp.autocast if self.transformer.dtype == torch.float16 and not self.use_lora else contextlib.nullcontext
        # Let's default to nullcontext for simplicity, user can override if needed for full model fine-tuning.
        return contextlib.nullcontext

    # --- Configuration and Saving/Loading ---

    def set_progress_bar_config(self, *args, **kwargs):
        self.flux_pipeline.set_progress_bar_config(*args, **kwargs)

    def save_pretrained(self, output_dir):
        """Saves the pipeline, including LoRA weights if used."""
        if self.use_lora:
            # FluxLoraLoaderMixin handles saving transformer LoRA layers
            self.flux_pipeline.save_lora_weights(save_directory=output_dir)
            logger.info(f"Saved LoRA weights to {output_dir}")

        else:
             # Save the entire pipeline (including potentially fine-tuned transformer)
             self.flux_pipeline.save_pretrained(output_dir)
             logger.info(f"Saved full pipeline state to {output_dir}")


    def get_trainable_layers(self):
        """Adds LoRA adapters if needed and returns the trainable model part."""
        if self.use_lora:
            # --- Add check for PEFT availability --- 
            if not is_peft_available():
                 raise ImportError(
                     "PEFT (`pip install peft`) is required to use LoRA adapters, but it's not installed."
                 )
            # ---------------------------------------
            # Check if adapters are already loaded/added
            if not hasattr(self.flux_pipeline.transformer, "peft_config") or getattr(self.flux_pipeline.transformer, "peft_config", None) is None:
                logger.info("Adding new LoRA adapters to the transformer.")

                lora_config = LoraConfig(
                    r=8, 
                    lora_alpha=8, 
                    init_lora_weights="gaussian",
                  
                    target_modules=[
                                    "attn.to_k",
                                    "attn.to_q",
                                    "attn.to_v",
                                    "attn.to_out.0",
                                    "attn.add_k_proj",
                                    "attn.add_q_proj",
                                    "attn.add_v_proj",
                                    "attn.to_add_out",
                                    "ff.net.0.proj",
                                    "ff.net.2",
                                    "ff_context.net.0.proj",
                                    "ff_context.net.2",
                                ],

                )
                try:
                     self.flux_pipeline.transformer.add_adapter(lora_config)
                     logger.info("Added LoRA adapters.")
                     # Ensure LoRA params are float32 for training stability
                     for param in self.flux_pipeline.transformer.parameters():
                         if param.requires_grad:
                             param.data = param.to(torch.float32)
                     logger.info("Set trainable LoRA parameters to float32.")
                except Exception as e:
                    logger.error(f"Failed to add LoRA adapter: {e}")
                    raise e
            else:
                 logger.info("LoRA adapters already present.")
                 # Still ensure trainable params are float32
                 for param in self.flux_pipeline.transformer.parameters():
                     if param.requires_grad:
                         param.data = param.to(torch.float32)

            # Return the transformer which now has trainable LoRA layers
            return self.flux_pipeline.transformer
        else:
            # Return the whole transformer for full fine-tuning
            logger.info("Using full transformer for training.")
            return self.flux_pipeline.transformer

    def save_checkpoint(self, models, weights, output_dir):
        """Saves the checkpoint during training."""
        if len(models) != 1:
            raise ValueError("Expected exactly one model (the transformer) from get_trainable_layers.")
        model_to_save = models[0]

        if self.use_lora and is_peft_available() and hasattr(model_to_save, "peft_config") and getattr(model_to_save, "peft_config", None) is not None:
            # Extract only the LoRA layers' state dictionary
            lora_state_dict = get_peft_model_state_dict(model_to_save)
            # Call the underlying save method, passing the specific layers
            self.flux_pipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=lora_state_dict # Pass the extracted LoRA layers
            )
            logger.info(f"Saved LoRA checkpoint to {output_dir}")
        elif not self.use_lora and isinstance(model_to_save, FluxTransformer2DModel):
            # Save the transformer model directly
            model_to_save.save_pretrained(os.path.join(output_dir, "transformer"))
            logger.info(f"Saved transformer checkpoint to {os.path.join(output_dir, 'transformer')}")
        else:
            raise ValueError(f"Cannot save checkpoint for model type {type(model_to_save)} with use_lora={self.use_lora}")


    def load_checkpoint(self, models, input_dir):
        """Loads the checkpoint during training."""
        if len(models) != 1:
            raise ValueError("Expected exactly one model (the transformer) to load checkpoint into.")
        model_to_load = models[0]

        if self.use_lora:
             # FluxLoraLoaderMixin handles loading into the transformer
            try:
                self.flux_pipeline.load_lora_weights(input_dir) # Loads into self.flux_pipeline.transformer
                 # Verify it loaded into the passed model instance (should be the same object)
                if model_to_load is not self.flux_pipeline.transformer:
                     logger.warning("LoRA loaded into pipeline's transformer, but a different model instance was passed to load_checkpoint.")
                 # Ensure float32 after loading if needed
                for param in model_to_load.parameters():
                     if param.requires_grad:
                         param.data = param.to(torch.float32)
                logger.info(f"Loaded LoRA checkpoint from {input_dir}")
            except Exception as e:
                 logger.error(f"Failed to load LoRA checkpoint from {input_dir}: {e}")
                 raise e
        elif not self.use_lora and isinstance(model_to_load, FluxTransformer2DModel):
            load_path = os.path.join(input_dir, "transformer")
            if not os.path.isdir(load_path):
                 raise ValueError(f"Transformer checkpoint directory not found at {load_path}")

            try:
                 loaded_transformer = FluxTransformer2DModel.from_pretrained(load_path)
                 model_to_load.load_state_dict(loaded_transformer.state_dict())
                 del loaded_transformer
                 logger.info(f"Loaded transformer checkpoint from {load_path}")
            except Exception as e:
                 logger.error(f"Failed to load transformer checkpoint from {load_path}: {e}")
                 raise e
        else:
             raise ValueError(f"Cannot load checkpoint for model type {type(model_to_load)} with use_lora={self.use_lora}")