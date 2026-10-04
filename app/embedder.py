"""Pretrained model wrapper: FashionCLIP for images + category, MiniLM for text."""
from __future__ import annotations

import numpy as np

CATEGORY_PROMPTS = {
    "bag": ["a photo of a handbag", "a photo of a backpack", "a photo of a purse",
            "a photo of a tote bag", "a photo of a travel bag"],
    "shoe": ["a photo of shoes", "a photo of sneakers", "a photo of sandals",
             "a photo of high heels", "a photo of boots", "a photo of slippers"],
    "clothing": ["a photo of a dress", "a photo of a shirt", "a photo of trousers",
                 "a photo of a jacket", "a photo of a skirt", "a photo of clothing"],
}


def _as_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"Unexpected model output type: {type(out)}")


class ClipEmbedder:
    def __init__(self, settings):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        dev = settings.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.device = dev

        self.model = CLIPModel.from_pretrained(settings.image_model).to(dev).eval()
        self.processor = CLIPProcessor.from_pretrained(settings.image_model)
        self.text_model = SentenceTransformer(settings.text_model, device=dev)

        self.image_dim = int(self.model.config.projection_dim)
        self.text_dim = int(self.text_model.get_sentence_embedding_dimension())
        self.categories = list(settings.categories)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self._cat_matrix = self._build_category_matrix()

    # ---------- images ----------
    def embed_images(self, images, batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        chunks = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                inputs = self.processor(images=images[i:i + batch_size], return_tensors="pt").to(self.device)
                feats = _as_tensor(self.model.get_image_features(**inputs))
                feats = torch.nn.functional.normalize(feats, dim=-1)
                chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks).astype("float32")

    # ---------- text ----------
    def embed_text(self, texts) -> np.ndarray:
        return self.text_model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

    # ---------- zero-shot category ----------
    def _clip_text(self, texts) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = _as_tensor(self.model.get_text_features(**inputs))
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype("float32")

    def _build_category_matrix(self) -> np.ndarray:
        rows = []
        for cat in self.categories:
            prompts = CATEGORY_PROMPTS.get(cat, [f"a photo of a {cat}"])
            v = self._clip_text(prompts).mean(axis=0)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows).astype("float32")

    def classify(self, image_embs: np.ndarray):
        """Return (category, confidence) averaged over all images of the product."""
        logits = self.logit_scale * image_embs @ self._cat_matrix.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        mean = probs.mean(axis=0)
        k = int(mean.argmax())
        return self.categories[k], float(mean[k])
