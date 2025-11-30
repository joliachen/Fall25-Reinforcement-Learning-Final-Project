from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    """Create a reward function measuring JPEG file size (incompressibility).

    The returned function converts images to JPEG at quality 95 and uses the
    resulting file size (in kilobytes) as the reward. Larger files correspond
    to less compressible images (more visual complexity / noise).

    Returns:
        Callable:
            A function `_fn(images, prompts, metadata)` that returns
            `(rewards, info)` where:

            * `images`:
                - `torch.Tensor` in `[0, 1]` of shape `(N, C, H, W)`, or
                - NumPy array/array-like of shape `(N, H, W, C)` in `[0, 255]`.
            * `prompts`: Unused, kept for API compatibility.
            * `metadata`: Unused, kept for API compatibility.
            * `rewards`: `np.ndarray` of shape `(N,)` with JPEG sizes in KB.
            * `info`: Empty dict.
    """
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    """Create a reward function for JPEG compressibility (negative file size).

    This is the negation of :func:`jpeg_incompressibility`. Smaller files
    (more compressible images) correspond to larger rewards.

    Returns:
        Callable:
            A function `_fn(images, prompts, metadata)` with the same inputs as
            in :func:`jpeg_incompressibility`, returning:

            * `rewards`: `np.ndarray` of shape `(N,)` with negative JPEG sizes
              in KB (higher is better).
            * `info`: Empty dict.
    """
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    """Create a reward function based on the aesthetic predictor.

    Uses :class:`ddpo_pytorch.aesthetic_scorer.AestheticScorer` with a CLIP
    backbone to produce scalar aesthetic scores for each image. Images are
    converted to uint8 tensors and passed directly to the scorer.

    Note:
        This function moves the scorer to CUDA by default and assumes GPU availability.

    Returns:
        Callable:
            A function `_fn(images, prompts, metadata)` that returns:
                * `rewards`: `torch.FloatTensor` of shape `(N,)` with aesthetic
                scores (higher is better).
                * `info`: Empty dict.
    """
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Reward from LLaVA QA accuracy with exact answer matching.

    Submits images and associated QA metadata to a LLaVA server and computes a
    per-image reward based on how many ground-truth answers are contained in
    the model's responses (no BERTScore, strict string containment).

    Metadata is expected to be an iterable of dicts with keys:
        * `"questions"`: list of strings (questions to ask about the image).
        * `"answers"`: list of strings (ground-truth answers in order).

    The server API is implemented in https://github.com/kvablack/LLaVA-server.

    Returns:
        Callable:
            A function `_fn(images, prompts, metadata)` that:

            * Sends images and associated questions to the LLaVA server.
            * Receives model responses per question.
            * Computes a per-image reward as the mean over questions of
              `1{answer is substring of response}`.

            It returns `(rewards, info)` where:
            * `rewards`: `np.ndarray` of shape `(N,)` with values in `[0, 1]`.
            * `info`: A dict containing:
                - `"answers"`: an array of model outputs for analysis.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Reward from LLaVA responses compared to prompts using BERTScore recall.

    Submits images and text prompts to a LLaVA server which:
        1. Generates a short description for each image.
        2. Compares the generated text to a templated answer based on the
           original prompt using BERTScore.
        3. Returns precision, recall, f1 and raw outputs.

    This reward uses the BERTScore *recall* as the scalar reward.

    The server API is implemented in https://github.com/kvablack/LLaVA-server.

    Returns:
        Callable:
            A function `_fn(images, prompts, metadata)` that:

            * Accepts `images` and `prompts` (metadata is ignored).
            * Sends JPEG-compressed images and text pairs to the server.
            * Returns `(rewards, info)` where:

            * `rewards`: `np.ndarray` of shape `(N,)` containing BERTScore
              recall values.
            * `info`: A dict with:
                - `"precision"`: array of precision scores.
                - `"f1"`: array of F1 scores.
                - `"outputs"`: array of generated text outputs.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
