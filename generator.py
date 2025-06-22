import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionUpscalePipeline
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import os
import time
from typing import Optional
from utils import Resolution, clean_compel_syntax

def setup_pipelines(
    model_id: str,
    torch_dtype: torch.dtype,
    memory_mode: str,
    use_upscaler: bool
) -> tuple[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, Optional[StableDiffusionUpscalePipeline], Compel]:
    """Sets up pipelines based on the selected memory mode and upscaler toggle."""
    
    # Determine the strategy based on memory mode
    final_memory_mode = memory_mode
    if memory_mode == 'auto':
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"ℹ️  Detected {vram_gb:.2f} GB of VRAM.")
            final_memory_mode = 'low' if vram_gb < 20 else 'high'
            print(f"✅ Auto-selected '{final_memory_mode}' memory mode.")
        else:
            print("⚠️  No CUDA device detected, defaulting to 'low' memory mode (CPU).")
            final_memory_mode = 'low'
    
    if torch_dtype == torch.float32:
        torch.set_float32_matmul_precision('high')

    # Load all pipelines to CPU first
    print("🔧 Loading base diffusion pipeline to CPU...")
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True, add_watermarker=False)
    
    print("🔧 Loading refiner pipeline to CPU...")
    refiner_pipeline = StableDiffusionXLImg2ImgPipeline(**base_pipeline.components)
    
    upscaler_pipeline = None
    if use_upscaler:
        print("🔧 Loading upscaler pipeline to CPU...")
        upscaler_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch_dtype, use_safetensors=True)

    # Now, move pipelines to GPU based on the chosen strategy
    graphics_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if final_memory_mode == 'high':
        print("🚀 Moving all models to GPU and compiling for maximum speed...")
        base_pipeline.to(graphics_device)
        refiner_pipeline.to(graphics_device)
        if upscaler_pipeline:
            upscaler_pipeline.to(graphics_device)
        base_pipeline.unet = torch.compile(base_pipeline.unet, mode="reduce-overhead", fullgraph=True)
    else: # 'low' memory mode (the new middle road)
        print("💾 Using memory-efficient mode. Moving only SDXL pipelines to GPU initially.")
        base_pipeline.to(graphics_device)
        refiner_pipeline.to(graphics_device)
        if upscaler_pipeline:
            # Leave the upscaler on the CPU for now
            upscaler_pipeline.to('cpu')

    print("🧠 Initializing Compel for prompt processing...")
    compel = Compel(
        tokenizer=[base_pipeline.tokenizer, base_pipeline.tokenizer_2],
        text_encoder=[base_pipeline.text_encoder, base_pipeline.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True])
        
    return base_pipeline, refiner_pipeline, upscaler_pipeline, compel


def generate(
    base_pipeline: StableDiffusionXLPipeline,
    refiner_pipeline: StableDiffusionXLImg2ImgPipeline,
    upscaler_pipeline: Optional[StableDiffusionUpscalePipeline],
    compel: Compel,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    guidance_scale: float,
    gen_resolution: Resolution,
    final_resolution: Resolution,
    initial_seed: int,
    output_dir: str
):
    """Generates images based on the provided prompts and parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if we need to do the memory-swap based on whether the upscaler is on the CPU
    is_low_mem_upscale = upscaler_pipeline is not None and upscaler_pipeline.device.type == 'cpu'
    
    graphics_device = base_pipeline.device
    
    print(f"ℹ️  Using Guidance Scale: {guidance_scale}")
    print("▶️  Processing and embedding prompts for SDXL...")
    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
    
    prompt_embeds = prompt_embeds.to(graphics_device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(graphics_device)
    negative_prompt_embeds = negative_prompt_embeds.to(graphics_device)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(graphics_device)

    print(f"🌱 Using initial seed: {initial_seed}")

    for i in range(num_images):
        print(f"\n--- 📸 Generating image {i + 1} of {num_images} ---")
        current_seed = initial_seed + i
        generator = torch.manual_seed(current_seed)
        print(f"🌱 Using seed for this image: {current_seed}")

        with torch.inference_mode():
            print(f"🎨 Generating base {gen_resolution.width}x{gen_resolution.height} latents...")
            base_image_latents = base_pipeline(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale, num_inference_steps=75,
                height=gen_resolution.height, width=gen_resolution.width,
                generator=generator, output_type="latent"
            ).images[0]

            print("✨ Refining and detailing the image...")
            refined_image = refiner_pipeline(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale, num_inference_steps=30, strength=0.25,
                image=base_image_latents, generator=generator
            ).images[0]

            if upscaler_pipeline:
                if is_low_mem_upscale:
                    print("💾 Moving SDXL pipelines to CPU to make space for upscaler...")
                    base_pipeline.to('cpu')
                    refiner_pipeline.to('cpu')
                    torch.cuda.empty_cache()
                    print("🚀 Moving upscaler to GPU...")
                    upscaler_pipeline.to(graphics_device)

                print("🧼 Cleaning prompts for the upscaler...")
                upscaler_prompt = clean_compel_syntax(prompt)
                upscaler_negative_prompt = clean_compel_syntax(negative_prompt)

                print(f"📈 Upscaling image with AI upscaler...")
                final_image = upscaler_pipeline(
                    prompt=upscaler_prompt,
                    negative_prompt=upscaler_negative_prompt,
                    image=refined_image,
                    num_inference_steps=20,
                    generator=generator
                ).images[0]

                if is_low_mem_upscale:
                    print("💾 Moving upscaler back to CPU...")
                    upscaler_pipeline.to('cpu')
                    torch.cuda.empty_cache()
                    print("🚀 Moving SDXL pipelines back to GPU for next image (if any)...")
                    base_pipeline.to(graphics_device)
                    refiner_pipeline.to(graphics_device)
            else:
                print("ℹ️  Upscaler disabled. Using image from refiner.")
                final_image = refined_image

        final_res_tuple = (final_resolution.width, final_resolution.height)
        print(f"💾 Resizing to final dimensions and saving: {final_res_tuple[0]}x{final_res_tuple[1]}...")
        final_image = final_image.resize(final_res_tuple, Image.LANCZOS)
        
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"image_{timestamp}_seed_{current_seed}.png")
        final_image.save(output_path)
        print(f"✅ Image saved to {output_path}")

    print(f"\n🎉 All {num_images} images generated successfully!")