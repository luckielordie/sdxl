import argparse
import torch
import random
import os
import time

from utils import Resolution, calculate_generation_dims, guidance_type, GUIDANCE_PRESETS, load_image
from generator import setup_pipelines, generate, generate_from_image

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion XL with an integrated latent upscaler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hardware & Model Settings
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Hugging Face model ID.")
    parser.add_argument("--dtype", default="float16", help="Torch dtype ('float16' or 'float32').")

    # Prompt Settings
    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    
    # Generation Settings
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument(
        "--guidance", type=guidance_type, default="medium",
        help=f"Guidance scale. Float or preset: {', '.join(GUIDANCE_PRESETS.keys())}"
    )
    parser.add_argument("--seed", type=int, default=-1, help="Initial seed. -1 for random.")
    parser.add_argument("--mode", type=str, default="text2img", choices=["text2img", "img2img"], help="Generation mode.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the source image for Img2Img.")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength for Img2Img generation.")
    
    # Image Dimension Settings
    parser.add_argument("--width", type=int, default=None, help="Final width. Defaults to source image width in img2img mode, or 1920 for text2img.")
    parser.add_argument("--height", type=int, default=None, help="Final height. Defaults to source image height in img2img mode, or 1080 for text2img.")

    # Output Settings
    parser.add_argument(
        "--output_dir", type=str, default=None, 
        help="Subdirectory within 'generated_images' to save outputs. Defaults to a timestamped folder."
    )

    args = parser.parse_args()

    if args.mode == "img2img" and not args.image_path:
        parser.error("--image_path is required for img2img mode.")

    BASE_OUTPUT_DIR = "generated_images"
    if not args.output_dir:
        args.output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{int(time.time())}")
    
    print(f"💾 Output will be saved to: {args.output_dir}")

    source_image = None
    if args.mode == 'img2img':
        source_image = load_image(args.image_path)
        if args.width is None or args.height is None:
            print("↔️ Using source image dimensions for output.")
            final_res = Resolution(width=source_image.width, height=source_image.height)
        else:
            final_res = Resolution(width=args.width, height=args.height)
    else:  # text2img
        width = args.width if args.width is not None else 1920
        height = args.height if args.height is not None else 1080
        final_res = Resolution(width=width, height=height)

    gen_res = calculate_generation_dims(final_res)
    
    initial_seed = args.seed if args.seed != -1 else random.randint(0, 2**32 - 1)
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    base_pipeline, refiner_pipeline, compel = setup_pipelines(
        args.model_id, torch_dtype
    )

    if args.mode == "text2img":
        generate(
            base_pipeline=base_pipeline,
            refiner_pipeline=refiner_pipeline,
            compel=compel,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            guidance_scale=args.guidance,
            initial_seed=initial_seed,
            gen_resolution=gen_res,
            final_resolution=final_res,
            output_dir=args.output_dir
        )
    elif args.mode == "img2img":
        generate_from_image(
            base_pipeline=base_pipeline,
            refiner_pipeline=refiner_pipeline,
            compel=compel,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            guidance_scale=args.guidance,
            initial_seed=initial_seed,
            gen_resolution=gen_res,
            final_resolution=final_res,
            output_dir=args.output_dir,
            source_image=source_image,
            strength=args.strength
        )

if __name__ == "__main__":
    main()