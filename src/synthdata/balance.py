"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.
"""

import random

from PIL import Image
from torchvision import transforms

# `augment()` runs before the training script's own TRAIN_TRANSFORM (crop, flip,
# mild colour jitter), which every photo gets. So only transforms that the
# pipeline lacks or applies weakly are added here. Hue stays fixed and the
# saturation change is mild, because coat colour and spots identify the breed.
AUGMENT = transforms.Compose(
    [
        transforms.RandomRotation(
            15, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomResizedCrop(
            256, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    ]
)


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    full_rounds, remainder = divmod(budget, len(minority_photos))
    chosen = list(minority_photos) * full_rounds
    chosen += rng.sample(minority_photos, remainder)
    rng.shuffle(chosen)
    return chosen


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. Typical choices are crops, flips, rotations, and
    colour changes from `torchvision.transforms`. Keep the cat recognisable.
    """
    return AUGMENT(image)
