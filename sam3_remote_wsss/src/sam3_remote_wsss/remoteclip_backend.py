from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image

from .config import ClassSpec, RemoteCLIPConfig


class RemoteCLIPPromptSelector:
    """Rank candidate prompts using RemoteCLIP image-text similarity.

    RemoteCLIP does not generate new text. In this project it scores a
    hand-written prompt bank and keeps the most relevant prompts for each tile.
    """

    def __init__(self, config: RemoteCLIPConfig) -> None:
        import open_clip
        import torch

        self.config = config
        self.open_clip = open_clip
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained="openai",
            device=config.device,
        )
        if config.checkpoint_path is not None:
            ckpt = torch.load(str(config.checkpoint_path), map_location="cpu")
            self.model.load_state_dict(ckpt)
        self.model = self.model.to(config.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.model_name)
        self._text_cache: dict[tuple[str, ...], object] = {}

    def select(
        self,
        image_rgb: np.ndarray,
        prompt_jobs: list[tuple[ClassSpec, str]],
    ) -> list[tuple[ClassSpec, str, float]]:
        prompts = [prompt for _, prompt in prompt_jobs]
        scores = self.score_prompts(image_rgb, prompts)
        grouped: dict[str, list[tuple[ClassSpec, str, float]]] = defaultdict(list)
        for (spec, prompt), score in zip(prompt_jobs, scores):
            grouped[spec.name].append((spec, prompt, float(score)))

        selected: list[tuple[ClassSpec, str, float]] = []
        for jobs in grouped.values():
            jobs = sorted(jobs, key=lambda item: item[2], reverse=True)
            if self.config.min_score is not None:
                jobs = [job for job in jobs if job[2] >= self.config.min_score]
            selected.extend(jobs[: self.config.top_k_per_class])
        return selected

    def score_prompts(self, image_rgb: np.ndarray, prompts: list[str]) -> np.ndarray:
        if not prompts:
            return np.empty((0,), dtype=np.float32)

        torch = self.torch
        pil_image = Image.fromarray(image_rgb, mode="RGB")
        image = self.preprocess(pil_image).unsqueeze(0).to(self.config.device)
        text_features = self._encode_text(prompts)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).squeeze(0)
        return scores.detach().float().cpu().numpy()

    def _encode_text(self, prompts: list[str]):
        key = tuple(prompts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        torch = self.torch
        tokens = self.tokenizer(prompts).to(self.config.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_cache[key] = text_features
        return text_features
