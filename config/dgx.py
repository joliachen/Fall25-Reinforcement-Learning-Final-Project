import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    """Build DDPO config for JPEG compressibility optimization on a DGX setup.

    This configuration is tuned for a DGX machine with 8 GPUs, using Stable
    Diffusion v1-4 and LoRA finetuning. It sets up a relatively small per-device
    batch size and uses gradient accumulation so that the effective number of
    samples and updates per epoch match the comments in the code.

    The reward function encourages images that are highly JPEG-compressible,
    using the "jpeg_compressibility" reward.

    Returns:
        ml_collections.ConfigDict: A configuration object with fields such as
        `pretrained`, `num_epochs`, `use_lora`, `sample`, `train`, `prompt_fn`,
        `reward_fn`, and `per_prompt_stat_tracking` set for the compressibility
        experiment.
    """
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    # set sample.batch_size = train.batch_size = 1 and multiply 
    # sample.num_batches_per_epoch and train.gradient_accumulation_steps accordingly
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 32

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 8

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def incompressibility():
    """Build DDPO config for JPEG incompressibility optimization on a DGX setup.

    This configuration is identical to :func:`compressibility` except for
    the reward function. Instead of favoring compressible images, it uses the
    `"jpeg_incompressibility"` reward to encourage harder-to-compress images.

    Returns:
        ml_collections.ConfigDict: A configuration object derived from
        :func:`compressibility` with `reward_fn` set to `"jpeg_incompressibility"`.
    """
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    """Build DDPO config for aesthetic score optimization on a DGX setup.

    This configuration starts from :func:`compressibility` and adjusts it for
    optimizing an aesthetic scoring model. It increases the number of training
    epochs and changes both the reward function and prompt distribution.

    Returns:
        ml_collections.ConfigDict: A configuration object customized for the
        aesthetic score optimization experiment.
    """
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def prompt_image_alignment():
    """Build DDPO config for prompt-image alignment (LLaVA + BERTScore) on a DGX.

    This configuration starts from :func:`compressibility` but adapts the
    sampling and training setup for a prompt-image alignment reward based on
    LLaVA and BERTScore.

    Returns:
        ml_collections.ConfigDict: A configuration object customized for the
        prompt-image alignment experiment with LLaVA + BERTScore rewards.
    """
    config = compressibility()

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def get_config(name):
    """Dispatch and build a DDPO configuration by name.

    This is a small helper used by the main training scripts to select one of
    the predefined DGX experiment configurations based on a string key.

    Args:
        name (str): Name of the configuration function to call. Typical values
            include:
            * `"compressibility"`
            * `"incompressibility"`
            * `"aesthetic"`
            * `"prompt_image_alignment"`

    Returns:
        ml_collections.ConfigDict: The configuration object returned by the
        corresponding function.

    Raises:
        KeyError: If `name` does not correspond to any global function in this module.
    """
    return globals()[name]()
