from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """Load lines from a text file, optionally resolving from the assets folder.

    This helper first checks whether `path` exists as given. If not, it looks
    for a file with the same name in the `ddpo_pytorch.assets` directory.

    Args:
        path: File path or filename to load. If it is not an existing path, it
            is interpreted relative to the `ddpo_pytorch.assets` package.

    Returns:
        A list of stripped lines (strings) read from the file.

    Raises:
        FileNotFoundError: If the file cannot be found either at `path` or in
        the assets directory.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    """Sample a random prompt from a text file.

    Lines are loaded via :func:`_load_lines`, then optionally sliced, and one
    line is chosen uniformly at random.

    Args:
        path: File path or filename to read prompts from. See `_load_lines`.
        low: Optional lower index for slicing the list of prompts (inclusive).
        high: Optional upper index for slicing the list of prompts (exclusive).

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair where `prompt` is a
        randomly chosen string and `metadata` is an empty dict.
    """
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    """Sample a random ImageNet class name (all 1000 classes).

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair, where `prompt` is a
        class name from `imagenet_classes.txt` and `metadata` is an empty dict.
    """
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    """Sample a random ImageNet animal class.

    Uses entries `[0:398]` from `imagenet_classes.txt`, which correspond to
    animal-related classes as in the original DDPO experiments.

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair.
    """
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    """Sample a random ImageNet dog class.

    Uses entries `[151:269]` from `imagenet_classes.txt`, which correspond to
    dog breeds.

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair.
    """
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    """Sample a random animal name from a small curated list.

    Uses `simple_animals.txt` in the assets folder.

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair.
    """
    return from_file("simple_animals.txt")


def nouns_activities(nouns_file, activities_file):
    """Sample a prompt of the form \"a/an NOUN ACTIVITY\".

    The noun and activity are sampled independently, and the article is chosen
    using `inflect` (e.g., `a cat`, `an owl`).

    Args:
        nouns_file: Filename or path to a text file containing one noun per
            line. Resolved via :func:`_load_lines`.
        activities_file: Filename or path to a text file containing one
            activity phrase per line.

    Returns:
        Tuple[str, dict]: A `(prompt, metadata)` pair where `prompt` is a
        string like `"a cat jumping"`, and `metadata` is an empty dict.
    """
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    """Sample a counting prompt and associated QA metadata.

    Prompts are of the form "three cats" where the number and noun are
    sampled randomly. Metadata includes a simple QA pair for use with
    vision-language rewards (e.g., LLaVA):

    * Questions:
        - "How many <plural_noun> are there in this image?"
        - "What animal is in this image?"
    * Answers:
        - `<number_in_words>`
        - `<singular_noun>`

    Args:
        nouns_file: Filename or path to a text file containing one singular
            noun per line.
        low: Minimum integer count (inclusive).
        high: Maximum integer count (inclusive).

    Returns:
        Tuple[str, dict]:
            * `prompt`: A string like `"three cats"`.
            * `metadata`: A dict with keys `"questions"` and `"answers"`,
              where each is a list of strings.
    """
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
