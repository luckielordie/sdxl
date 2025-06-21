import argparse
import torch
import random
import os
import time

from utils import Resolution, calculate_generation_dims, guidance_type, GUIDANCE_PRESETS
from generator import setup_pipelines, generate

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion XL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Hugging Face model ID.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda', 'cpu').")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype ('float16' or 'float32').")

    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    
    parser.add_argument(
        "--guidance", type=guidance_type, default="medium",
        help=f"Guidance scale. Float or preset: {', '.join(GUIDANCE_PRESETS.keys())}"
    )
    parser.add_argument("--seed", type=int, default=-1, help="Initial seed. -1 for random.")
    
    parser.add_argument("--width", type=int, default=1920, help="Final width of the output image.")
    parser.add_argument("--height", type=int, default=1080, help="Final height of the output image.")

    parser.add_argument(
        "--output_dir", type=str, default=None, 
        help="Subdirectory within 'generated_images' to save outputs. Defaults to a timestamped folder."
    )

    args = parser.parse_args()

    BASE_OUTPUT_DIR = "generated_images"
    if args.output_dir:
        final_output_dir = os.path.join(BASE_OUTPUT_DIR, args.output_dir)
    else:
        timestamp = int(time.time())
        final_output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
    
    print(f"💾 Output will be saved to: {final_output_dir}")

    final_res = Resolution(width=args.width, height=args.height)
    gen_res = calculate_generation_dims(final_res)
    
    initial_seed = args.seed if args.seed != -1 else random.randint(0, 2**32 - 1)

    graphics_device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    base_pipeline, refiner_pipeline, upscaler_pipeline, compel = setup_pipelines(args.model_id, graphics_device, torch_dtype)

    generate(
        base_pipeline=base_pipeline,
        refiner_pipeline=refiner_pipeline,
        upscaler_pipeline=upscaler_pipeline,
        compel=compel,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images=args.num_images,
        guidance_scale=args.guidance,
        initial_seed=initial_seed,
        gen_resolution=gen_res,
        final_resolution=final_res,
        output_dir=final_output_dir
    )

if __name__ == "__main__":
    main()