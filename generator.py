import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import os
import time
from utils import Resolution

def setup_pipelines(
    model_id: str,
    torch_dtype: torch.dtype,
) -> tuple[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, Compel]:
    """Sets up pipelines for manual VRAM management."""    
    if torch_dtype == torch.float32:
        torch.set_float32_matmul_precision('high')

    # In 'high' mode, all models go to GPU. In 'low' mode, they stay on CPU for now.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Loading all pipelines to {device}...")

    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True, add_watermarker=False).to(device)
    
    refiner_pipeline = StableDiffusionXLImg2ImgPipeline(**base_pipeline.components).to(device)

    print("🔥 Compiling UNets for maximum speed...")
    base_pipeline.unet = torch.compile(base_pipeline.unet, mode="reduce-overhead", fullgraph=True)
    refiner_pipeline.unet = torch.compile(refiner_pipeline.unet, mode="reduce-overhead", fullgraph=True)

    print("🧠 Initializing Compel for prompt processing...")
    compel = Compel(
        tokenizer=[base_pipeline.tokenizer, base_pipeline.tokenizer_2],
        text_encoder=[base_pipeline.text_encoder, base_pipeline.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True])
        
    return base_pipeline, refiner_pipeline, compel

def generate(
    base_pipeline: StableDiffusionXLPipeline,
    refiner_pipeline: StableDiffusionXLImg2ImgPipeline,
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
    """Generates images with manual pipeline management for memory efficiency."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"ℹ️  Using Guidance Scale: {guidance_scale}")
    print("▶️  Processing and embedding prompts for SDXL...")
    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)

    print(f"🌱 Using initial seed: {initial_seed}")

    for i in range(num_images):
        print(f"\n--- 📸 Generating image {i + 1} of {num_images} ---")
        current_seed = initial_seed + i
        generator = torch.manual_seed(current_seed)
        print(f"🌱 Using seed for this image: {current_seed}")

        with torch.inference_mode():            
            print(f"🎨 Generating base {gen_resolution.width}x{gen_resolution.height} latents...")
            base_image_latents = base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale, num_inference_steps=75,
                height=gen_resolution.height, width=gen_resolution.width,
                generator=generator, output_type="latent"
            ).images[0]

            print("✨ Refining and detailing the image...")
            refined_image = refiner_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale, num_inference_steps=30, strength=0.25,
                image=base_image_latents, generator=generator
            ).images[0]

        final_res_tuple = (final_resolution.width, final_resolution.height)
        print(f"💾 Resizing to final dimensions and saving: {final_res_tuple[0]}x{final_res_tuple[1]}...")
        
        final_image = refined_image.resize(final_res_tuple, Image.LANCZOS)
        
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"image_{timestamp}_seed_{current_seed}.png")
        final_image.save(output_path)
        print(f"✅ Image saved to {output_path}")

    print(f"\n🎉 All {num_images} images generated successfully!")
