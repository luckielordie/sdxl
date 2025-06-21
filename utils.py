import math
import argparse
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Resolution:
    """A simple, immutable dataclass to hold width and height."""
    width: int
    height: int

GUIDANCE_PRESETS = {
    "artistic": 4.0,  # More creative, less prompt-adherent
    "low": 5.0,
    "medium": 7.5,    # Balanced default
    "high": 11.0,
    "strict": 12.5,   # Very strong prompt adherence
}

def clean_compel_syntax(prompt: str) -> str:
    """
    Removes or simplifies Compel-specific syntax from a prompt string
    to make it compatible with standard pipelines.
    """
    # Handle [from:to:when] syntax by taking the 'to' part.
    prompt = re.sub(r"\[[^\]:]*:([^\]:]*):[^\]]*\]", r"\1", prompt)

    # Handle (word:weight) syntax by just taking the word.
    prompt = re.sub(r"\(([^:\)]+):[0-9.]+\)", r"\1", prompt)

    # Remove all remaining weighting symbols like (), +, -
    prompt = re.sub(r"[\(\)\+\-]+", " ", prompt)

    # Finally, clean up any extra whitespace.
    prompt = re.sub(r" +", " ", prompt).strip()

    return prompt

def guidance_type(value: str) -> float:
    """Custom type for argparse to handle both preset strings and float values."""
    lower_value = value.lower()
    if lower_value in GUIDANCE_PRESETS:
        return GUIDANCE_PRESETS[lower_value]
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid guidance value: '{value}'. "
            f"Must be a float (e.g., 8.0) or a preset: {', '.join(GUIDANCE_PRESETS.keys())}"
        )

def calculate_generation_dims(
    final_resolution: Resolution, 
    target_area: int = 1024*1024, 
    multiple_of: int = 64
) -> Resolution:
    """Calculates generation dimensions that maintain aspect ratio while targeting a specific area."""
    aspect_ratio = final_resolution.width / final_resolution.height
    
    gen_height_ideal = math.sqrt(target_area / aspect_ratio)
    gen_width_ideal = aspect_ratio * gen_height_ideal

    gen_width = int(round(gen_width_ideal / multiple_of) * multiple_of)
    gen_height = int(round(gen_height_ideal / multiple_of) * multiple_of)
    
    gen_width = max(multiple_of, gen_width)
    gen_height = max(multiple_of, gen_height)

    print(f"📐 Requested final dimensions: {final_resolution.width}x{final_resolution.height}")
    calculated_res = Resolution(width=gen_width, height=gen_height)
    print(f"🤖 Calculated generation dimensions (closest to {target_area/1e6:.2f}MP): {calculated_res.width}x{calculated_res.height}")
    
    return calculated_res