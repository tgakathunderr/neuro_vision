"""Procedural Dataset Generator for Few-Shot Visual Object Recognition & Generalization."""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASSES = ["Cat", "Car", "Airplane", "Tree", "Coffee Cup", "Star"]


def _draw_cat(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws a recognizable feline object."""
    r = size // 2
    if not alt:
        # Canonical cat profile: body, head, triangular ears, tail
        # Body
        draw.ellipse([cx - r, cy - int(r * 0.4), cx + int(r * 0.6), cy + int(r * 0.8)], fill=240, outline=255)
        # Head
        draw.ellipse([cx + int(r * 0.2), cy - int(r * 0.8), cx + int(r * 0.9), cy - int(r * 0.1)], fill=240, outline=255)
        # Ears
        draw.polygon([(cx + int(r * 0.3), cy - int(r * 0.7)), (cx + int(r * 0.4), cy - int(r * 1.1)), (cx + int(r * 0.55), cy - int(r * 0.7))], fill=255)
        draw.polygon([(cx + int(r * 0.65), cy - int(r * 0.7)), (cx + int(r * 0.8), cy - int(r * 1.1)), (cx + int(r * 0.9), cy - int(r * 0.6))], fill=255)
        # Tail
        draw.arc([cx - int(r * 1.2), cy - int(r * 0.3), cx - int(r * 0.6), cy + int(r * 0.7)], start=90, end=270, fill=255, width=3)
    else:
        # Alternate exemplar: Frontal cat face with whiskers
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=240, outline=255)
        # Left ear
        draw.polygon([(cx - r, cy - int(r * 0.4)), (cx - int(r * 0.8), cy - int(r * 1.2)), (cx - int(r * 0.2), cy - int(r * 0.8))], fill=255)
        # Right ear
        draw.polygon([(cx + int(r * 0.2), cy - int(r * 0.8)), (cx + int(r * 0.8), cy - int(r * 1.2)), (cx + r, cy - int(r * 0.4))], fill=255)
        # Eyes
        draw.ellipse([cx - int(r * 0.5), cy - int(r * 0.2), cx - int(r * 0.2), cy + int(r * 0.1)], fill=20)
        draw.ellipse([cx + int(r * 0.2), cy - int(r * 0.2), cx + int(r * 0.5), cy + int(r * 0.1)], fill=20)
        # Whiskers
        draw.line([cx - int(r * 0.2), cy + int(r * 0.3), cx - int(r * 1.1), cy + int(r * 0.2)], fill=255, width=2)
        draw.line([cx - int(r * 0.2), cy + int(r * 0.4), cx - int(r * 1.0), cy + int(r * 0.5)], fill=255, width=2)
        draw.line([cx + int(r * 0.2), cy + int(r * 0.3), cx + int(r * 1.1), cy + int(r * 0.2)], fill=255, width=2)
        draw.line([cx + int(r * 0.2), cy + int(r * 0.4), cx + int(r * 1.0), cy + int(r * 0.5)], fill=255, width=2)


def _draw_car(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws a vehicle / car profile."""
    r = size // 2
    if not alt:
        # Sedan profile
        # Chassis body
        draw.rectangle([cx - int(r * 1.1), cy, cx + int(r * 1.1), cy + int(r * 0.5)], fill=240, outline=255)
        # Cabin roof
        draw.polygon([
            (cx - int(r * 0.7), cy),
            (cx - int(r * 0.4), cy - int(r * 0.6)),
            (cx + int(r * 0.4), cy - int(r * 0.6)),
            (cx + int(r * 0.7), cy),
        ], fill=240, outline=255)
        # Wheels
        wr = int(r * 0.28)
        draw.ellipse([cx - int(r * 0.7) - wr, cy + int(r * 0.4) - wr, cx - int(r * 0.7) + wr, cy + int(r * 0.4) + wr], fill=20, outline=255, width=2)
        draw.ellipse([cx + int(r * 0.7) - wr, cy + int(r * 0.4) - wr, cx + int(r * 0.7) + wr, cy + int(r * 0.4) + wr], fill=20, outline=255, width=2)
    else:
        # Alternate exemplar: Truck / SUV boxy profile
        draw.rectangle([cx - int(r * 1.1), cy - int(r * 0.7), cx, cy + int(r * 0.5)], fill=240, outline=255)
        draw.rectangle([cx, cy, cx + int(r * 1.1), cy + int(r * 0.5)], fill=240, outline=255)
        # Wheels
        wr = int(r * 0.28)
        draw.ellipse([cx - int(r * 0.6) - wr, cy + int(r * 0.4) - wr, cx - int(r * 0.6) + wr, cy + int(r * 0.4) + wr], fill=20, outline=255, width=2)
        draw.ellipse([cx + int(r * 0.7) - wr, cy + int(r * 0.4) - wr, cx + int(r * 0.7) + wr, cy + int(r * 0.4) + wr], fill=20, outline=255, width=2)


def _draw_airplane(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws an aircraft with fuselage and wings."""
    r = size // 2
    if not alt:
        # Jet top-down view
        # Fuselage
        draw.ellipse([cx - int(r * 0.25), cy - int(r * 1.1), cx + int(r * 0.25), cy + int(r * 1.1)], fill=240, outline=255)
        # Wings
        draw.polygon([
            (cx - int(r * 1.1), cy + int(r * 0.1)),
            (cx + int(r * 1.1), cy + int(r * 0.1)),
            (cx + int(r * 0.2), cy - int(r * 0.3)),
            (cx - int(r * 0.2), cy - int(r * 0.3)),
        ], fill=240, outline=255)
        # Tail stabilizer
        draw.polygon([
            (cx - int(r * 0.6), cy + int(r * 0.9)),
            (cx + int(r * 0.6), cy + int(r * 0.9)),
            (cx, cy + int(r * 0.6)),
        ], fill=240, outline=255)
    else:
        # Alternate: Airplane side silhouette
        draw.polygon([
            (cx - int(r * 1.1), cy),
            (cx + int(r * 1.1), cy - int(r * 0.1)),
            (cx + int(r * 0.9), cy + int(r * 0.2)),
            (cx - int(r * 0.9), cy + int(r * 0.2)),
        ], fill=240, outline=255)
        # Wing tilted
        draw.polygon([(cx - int(r * 0.2), cy), (cx - int(r * 0.5), cy - int(r * 0.8)), (cx + int(r * 0.2), cy)], fill=255)
        # Tail fin
        draw.polygon([(cx - int(r * 1.0), cy), (cx - int(r * 1.1), cy - int(r * 0.6)), (cx - int(r * 0.7), cy)], fill=255)


def _draw_tree(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws a tree with trunk and foliage."""
    r = size // 2
    if not alt:
        # Deciduous tree (round foliage)
        # Trunk
        draw.rectangle([cx - int(r * 0.2), cy, cx + int(r * 0.2), cy + int(r * 1.0)], fill=160, outline=255)
        # Foliage cluster
        draw.ellipse([cx - int(r * 0.8), cy - int(r * 1.0), cx + int(r * 0.8), cy + int(r * 0.2)], fill=240, outline=255)
        draw.ellipse([cx - int(r * 0.9), cy - int(r * 0.6), cx, cy + int(r * 0.2)], fill=230)
        draw.ellipse([cx, cy - int(r * 0.6), cx + int(r * 0.9), cy + int(r * 0.2)], fill=230)
    else:
        # Coniferous / Pine tree (triangular tiered foliage)
        # Trunk
        draw.rectangle([cx - int(r * 0.15), cy + int(r * 0.4), cx + int(r * 0.15), cy + int(r * 1.0)], fill=160)
        # Tier 3 (bottom)
        draw.polygon([(cx, cy - int(r * 0.2)), (cx - int(r * 0.9), cy + int(r * 0.5)), (cx + int(r * 0.9), cy + int(r * 0.5))], fill=240, outline=255)
        # Tier 2 (middle)
        draw.polygon([(cx, cy - int(r * 0.6)), (cx - int(r * 0.75), cy), (cx + int(r * 0.75), cy)], fill=240, outline=255)
        # Tier 1 (top)
        draw.polygon([(cx, cy - int(r * 1.1)), (cx - int(r * 0.55), cy - int(r * 0.4)), (cx + int(r * 0.55), cy - int(r * 0.4))], fill=240, outline=255)


def _draw_cup(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws a coffee mug / cup."""
    r = size // 2
    if not alt:
        # Standard cylindrical coffee mug with handle
        draw.rectangle([cx - int(r * 0.6), cy - int(r * 0.5), cx + int(r * 0.5), cy + int(r * 0.7)], fill=240, outline=255)
        # Rounded bottom
        draw.ellipse([cx - int(r * 0.6), cy + int(r * 0.5), cx + int(r * 0.5), cy + int(r * 0.8)], fill=240)
        # Handle
        draw.arc([cx + int(r * 0.2), cy - int(r * 0.4), cx + int(r * 1.0), cy + int(r * 0.5)], start=270, end=90, fill=255, width=4)
        # Steam line
        draw.line([cx - int(r * 0.2), cy - int(r * 0.7), cx - int(r * 0.1), cy - int(r * 1.1)], fill=200, width=2)
        draw.line([cx + int(r * 0.2), cy - int(r * 0.7), cx + int(r * 0.3), cy - int(r * 1.1)], fill=200, width=2)
    else:
        # Tea cup with saucer
        # Saucer
        draw.ellipse([cx - int(r * 1.1), cy + int(r * 0.5), cx + int(r * 1.1), cy + int(r * 0.85)], fill=220, outline=255)
        # Cup bowl
        draw.chord([cx - int(r * 0.7), cy - int(r * 0.4), cx + int(r * 0.7), cy + int(r * 0.6)], start=0, end=180, fill=240, outline=255)
        # Handle
        draw.arc([cx + int(r * 0.4), cy - int(r * 0.3), cx + int(r * 1.0), cy + int(r * 0.3)], start=270, end=90, fill=255, width=3)


def _draw_star(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, alt: bool = False):
    """Draws a 5-pointed or 4-pointed geometric star."""
    r = size // 2
    points = []
    n_points = 4 if alt else 5
    inner_r = r * 0.4

    for i in range(n_points * 2):
        angle = i * math.pi / n_points - math.pi / 2.0
        rad = r if (i % 2 == 0) else inner_r
        px = cx + rad * math.cos(angle)
        py = cy + rad * math.sin(angle)
        points.append((px, py))

    draw.polygon(points, fill=240, outline=255)


DRAW_FUNCS = {
    "Cat": _draw_cat,
    "Car": _draw_car,
    "Airplane": _draw_airplane,
    "Tree": _draw_tree,
    "Coffee Cup": _draw_cup,
    "Star": _draw_star,
}


def render_object_image(
    class_name: str,
    variation: str = "prototype",
    canvas_size: Tuple[int, int] = (64, 64),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Renders an object image with procedural variations:
    - prototype: Canonical reference image for 1-shot learning.
    - rotated: Rotated ±25° to ±40°.
    - scaled: Scaled down (70%) or zoomed in (125%).
    - translated: Shifted off-center by (dx, dy).
    - noisy: Gaussian noise & contrast degradation.
    - alternate: Completely distinct structural silhouette/exemplar.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    w, h = canvas_size
    img = Image.new("L", (w, h), color=15)  # Dark background
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    base_size = int(min(w, h) * 0.65)
    is_alt = (variation == "alternate")

    # 1. Base shape drawing
    draw_fn = DRAW_FUNCS.get(class_name, _draw_cat)

    if variation == "prototype":
        draw_fn(draw, cx, cy, base_size, alt=False)

    elif variation == "rotated":
        # Draw on larger canvas and rotate
        angle = random.choice([-35, -25, 25, 35])
        big_size = (w * 2, h * 2)
        big_img = Image.new("L", big_size, color=15)
        big_draw = ImageDraw.Draw(big_img)
        draw_fn(big_draw, w, h, int(base_size * 1.5), alt=False)
        rotated = big_img.rotate(angle, resample=Image.Resampling.BILINEAR)
        # Crop back to center
        img = rotated.crop((w // 2, h // 2, w // 2 + w, h // 2 + h))

    elif variation == "scaled":
        scale_factor = random.choice([0.70, 1.25])
        new_size = int(base_size * scale_factor)
        draw_fn(draw, cx, cy, new_size, alt=False)

    elif variation == "translated":
        dx = random.choice([-10, -8, 8, 10])
        dy = random.choice([-10, -6, 6, 10])
        draw_fn(draw, cx + dx, cy + dy, base_size, alt=False)

    elif variation == "noisy":
        # Draw base prototype
        draw_fn(draw, cx, cy, base_size, alt=False)
        arr = np.asarray(img, dtype=np.float32)
        # Add speckle Gaussian noise
        noise = np.random.normal(0.0, 35.0, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_arr)
        # Slight blur / contrast drop
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    elif variation == "alternate":
        draw_fn(draw, cx, cy, base_size, alt=True)

    else:
        draw_fn(draw, cx, cy, base_size, alt=False)

    return np.asarray(img, dtype=np.float32) / 255.0


def generate_benchmark_dataset() -> Dict[str, Any]:
    """
    Constructs the standardized benchmark dataset:
    - 6 Prototypes (1 per class) for 1-shot learning.
    - 30 Unseen Test Images (5 distinct unseen transformation variations per class).
    """
    prototypes = {}
    for cls in CLASSES:
        prototypes[cls] = render_object_image(cls, variation="prototype", seed=42)

    test_samples = []
    variations = ["rotated", "scaled", "translated", "noisy", "alternate"]

    for cls in CLASSES:
        for idx, var in enumerate(variations):
            seed = 100 + CLASSES.index(cls) * 10 + idx
            img_arr = render_object_image(cls, variation=var, seed=seed)
            test_samples.append({
                "class": cls,
                "variation": var,
                "image": img_arr,
                "label_idx": CLASSES.index(cls),
            })

    return {
        "classes": CLASSES,
        "prototypes": prototypes,
        "test_samples": test_samples,
    }
