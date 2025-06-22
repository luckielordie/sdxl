# Stable Diffusion XL Image Generator

This is a powerful and user-friendly command-line tool for generating high-quality images using Stable Diffusion XL. It is designed for both ease of use and advanced control, featuring optional AI upscaling, intelligent memory management, and a modular codebase.

## Features

- **High-Quality Generation**: Utilizes the base SDXL model followed by a refiner for detailed, high-resolution images.
- **Optional AI Upscaling**: Features an integrated `stabilityai/stable-diffusion-x4-upscaler` pipeline that can be toggled on to intelligently increase image resolution and detail, going beyond a simple resize.
- **Intelligent Memory Management**: Automatically detects your system's VRAM and selects the best operational mode. On high-VRAM systems, it loads all models to the GPU for maximum speed. on low-VRAM systems, it uses CPU offloading to ensure the script can run without crashing.
- **Flexible Guidance Control**: Use simple presets (`low`, `medium`, `high`) or specify an exact float value for precise control over prompt adherence.
- **Reproducibility**: Set a specific seed to generate the same image again, or let the script choose a random one.
- **Optimized Performance**: In "high memory" mode, it leverages `torch.compile` for the fastest possible generation on compatible hardware.
- **Clean & Modular Code**: The project is broken into logical files (`main.py`, `generator.py`, `utils.py`) for easy reading and extension.

## Setup & Installation

This project is managed with [Poetry](https://python-poetry.org/).

1.  **Clone or download the project files** into a single directory.

2.  **Install Poetry**: Follow the [official installation guide](https://python-poetry.org/docs/#installation) for your operating system.

3.  **Install Dependencies**: Navigate to the project's root directory in your terminal and run:
    ```bash
    poetry install
    ```
    This command will create a virtual environment and install all necessary dependencies, including PyTorch with CUDA support.

## Local Usage

All commands are run from your terminal in the project's root directory. The main entry point is `main.py`. The only required argument is `--prompt`.

### Basic Example

This command will run the base and refiner pipelines in the most memory-efficient way.

```bash
python main.py --prompt "a dramatic photo of a majestic lion in the savanna, cinematic lighting, 8k"
````

### High-Quality Example with AI Upscaler

This command enables the AI upscaler for the best possible output quality. It's recommended to run this on a machine with sufficient VRAM or with the `auto` or `low` memory modes.

```bash
python main.py \
    --prompt "an astronaut lounging in a tropical resort on mars, cinematic, 4k" \
    --guidance high \
    --use_upscaler
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--prompt` | `string` | **(Required)** | The positive prompt describing the image you want. |
| `--negative_prompt` | `string` | `""` | The negative prompt, describing what to avoid. |
| `--model_id` | `string` | `stabilityai/stable-diffusion-xl-base-1.0`| The Hugging Face model ID for the base SDXL model. |
| `--use_upscaler` | `flag` | `False` | Add this flag to enable the AI upscaler pipeline for higher quality. |
| `--memory_mode` | `string` | `auto` | Memory mode: `auto`, `high` (VRAM-only), `low` (CPU offload). |
| `--guidance` | `preset/float` | `medium` | Prompt adherence. Can be a float (`8.2`) or a preset: `artistic`, `low`, `medium`, `high`, `strict`. |
| `--num_images` | `int` | `1` | The number of images to generate in a single run. |
| `--seed` | `int` | `-1` | The starting seed. `-1` means a random seed will be chosen. |
| `--width` | `int` | `1920` | The **final width** of the output image. |
| `--height` | `int` | `1080` | The **final height** of the output image. |
| `--output_dir` | `string` | `None` | Subdirectory within `generated_images` to save to. Defaults to a timestamped folder. |
| `--dtype` | `string` | `float16` | The torch data type (`float16` for performance, `float32` for precision). |

### Advanced Examples

**1. Generate a portrait for a phone screen with the AI upscaler:**

```bash
python main.py \
    --prompt "full body portrait of a sci-fi queen on a throne, intricate armor, cinematic" \
    --negative_prompt "blurry, ugly, deformed, cartoon" \
    --width 1080 \
    --height 1920 \
    --guidance high \
    --use_upscaler
```

*The script will generate at an optimal base resolution, refine the image, and then use the AI upscaler for the final dimensions.*

**2. Generate 5 images with a specific seed, forcing high-performance mode:**

This is for machines with plenty of VRAM (e.g., \>20GB).

```bash
python main.py \
    --prompt "a cozy cabin in a winter forest at night, stars visible, glowing windows" \
    --num_images 5 \
    --seed 42 \
    --output_dir "winter_cabins" \
    --use_upscaler \
    --memory_mode high
```

*This will generate 5 images with seeds 42-46, with all models loaded onto the GPU for maximum speed.*
