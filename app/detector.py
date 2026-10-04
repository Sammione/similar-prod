"""
Core logic: when a vendor uploads a product, find products from OTHER vendors
that look like it and raise a risk alarm.

Three signals are combined:
  1. Image similarity   - FashionCLIP embeddings (catches the same item shot differently)
  2. Text similarity    - MiniLM embeddings of title + description
  3. Photo fingerprint  - perceptual hash (catches a copied/re-saved/resized photo)
Price closeness is a small extra signal.
"""
from __future__ import annotations

import io
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import imagehash
import numpy as np
from PIL import Image, ImageOps

from config import Settings
from .db import Database
from .vector_index import VectorIndex

ImageInput = Union[Image.Image, bytes, str, Path]
ACTIONS = {"HIGH": "hold_for_review", "MEDIUM": "publish_and_flag", "LOW": "publish"}


def load_image(src: ImageInput) -> Image.Image:
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _product_text(title: str, description: str) -> str:
    return f"{title or ''}. {description or ''}".strip(". ")


def _price_similarity(a, b) -> Optional[float]:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0:
        return None
    return 1.0 - min(1.0, abs(a - b) / max(a, b))


@dataclass
class Match:
    product_id: int
    vendor_id: str
    title: str
    category: str
    score: float
    level: str
    image_similarity: float
    text_similarity: float
    price_similarity: Optional[float]
    photo_hash_distance: Optional[int]
    reasons: list = field(default_factory=list)


@dataclass
class CheckResult:
    category: str
    category_confidence: float
    category_warning: Optional[str]
    risk_level: str            # HIGH | MEDIUM | LOW
    alarm: bool
    recommended_action: str    # hold_for_review | publish_and_flag | publish
    matches: list

    def to_dict(self):
        return asdict(self)


@dataclass
class _Prepared:
    img_embs: np.ndarray
    txt_emb: np.ndarray
    hashes: list
    category: str
    predicted_category: str
    category_confidence: float
    category_warning: Optional[str]


class SimilarityDetector:
    def __init__(self, settings: Settings, embedder=None):
        self.s = settings
        self.data_dir = Path(settings.data_dir)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "catalog.db")

        if embedder is None:
            from .embedder import ClipEmbedder
            embedder = ClipEmbedder(settings)
        self.emb = embedder

        self.img_idx = {c: VectorIndex(embedder.image_dim, self.data_dir / f"img_{c}.faiss")
                        for c in settings.categories}
        self.txt_idx = {c: VectorIndex(embedder.text_dim, self.data_dir / f"txt_{c}.faiss")
                        for c in settings.categories}
        # in-memory photo fingerprints: (image_id, product_id, ImageHash)
        self._hashes = [(r["id"], r["product_id"], imagehash.hex_to_hash(r["phash"]))
                        for r in self.db.all_hashes()]
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ public
    def check(self, vendor_id: str, title: str, description: str = "", price=None,
              images: Sequence[ImageInput] = (), declared_category: Optional[str] = None) -> CheckResult:
        """Score a product against the catalogue WITHOUT saving it."""
        imgs = self._load(images)
        with self._lock:
            p = self._prepare(imgs, title, description, declared_category)
            return self._result(p, self._find_matches(vendor_id, price, p))

    def register(self, vendor_id: str, title: str, description: str = "", price=None,
                 images: Sequence[ImageInput] = (), declared_category: Optional[str] = None,
                 external_id: Optional[str] = None, create_alerts: bool = True, save: bool = True):
        """Score a product, add it to the catalogue, and store any alerts. Returns (product_id, CheckResult)."""
        imgs = self._load(images)
        with self._lock:
            p = self._prepare(imgs, title, description, declared_category)
            result = self._result(p, self._find_matches(vendor_id, price, p))

            pid = self.db.insert_product(vendor_id, title, description, price, p.category,
                                         p.predicted_category, external_id)
            image_ids = []
            for k, (img, h) in enumerate(zip(imgs, p.hashes)):
                path = self.images_dir / f"{pid}_{k}.jpg"
                img.save(path, "JPEG", quality=90)
                iid = self.db.insert_image(pid, str(path), str(h))
                image_ids.append(iid)
                self._hashes.append((iid, pid, h))
            self.img_idx[p.category].add(image_ids, p.img_embs)
            self.txt_idx[p.category].add([pid], p.txt_emb[None, :])

            if create_alerts:
                for m in result.matches:
                    self.db.insert_alert(pid, vendor_id, asdict(m))
            if save:
                self.save()
        return pid, result

    def remove_product(self, product_id: int) -> bool:
        with self._lock:
            prod = self.db.get_product(product_id)
            if not prod or not prod["active"]:
                return False
            image_ids = self.db.image_ids_for_product(product_id)
            self.img_idx[prod["category"]].remove(image_ids)
            self.txt_idx[prod["category"]].remove([product_id])
            self.db.deactivate_product(product_id)
            self._hashes = [t for t in self._hashes if t[1] != product_id]
            self.save()
        return True

    def save(self):
        with self._lock:
            for idx in list(self.img_idx.values()) + list(self.txt_idx.values()):
                idx.save()

    def stats(self):
        return {
            "active_products": self.db.count_products(),
            "images_per_category": {c: len(i) for c, i in self.img_idx.items()},
            "open_alerts": len(self.db.list_alerts(status="open", limit=100000)),
        }

    # ----------------------------------------------------------------- internal
    @staticmethod
    def _load(images):
        imgs = [load_image(x) for x in images]
        if not imgs:
            raise ValueError("At least one product image is required")
        return imgs

    def _prepare(self, imgs, title, description, declared_category) -> _Prepared:
        img_embs = self.emb.embed_images(imgs)
        txt_emb = self.emb.embed_text([_product_text(title, description)])[0]
        hashes = [imagehash.phash(im) for im in imgs]
        predicted, conf = self.emb.classify(img_embs)

        category, warning = predicted, None
        if declared_category:
            d = declared_category.strip().lower()
            if d not in self.s.categories:
                raise ValueError(f"Unknown category '{declared_category}'. Use one of {list(self.s.categories)}")
            category = d
            if d != predicted and conf >= 0.6:
                warning = f"Declared '{d}' but images look like '{predicted}' ({conf:.0%} confidence)"
        return _Prepared(img_embs, txt_emb, hashes, category, predicted, round(conf, 4), warning)

    def _find_matches(self, vendor_id, price, p: _Prepared):
        s = self.s
        cand: dict = {}  # product_id -> best photo-hash distance or None

        # Search both declared and predicted category, so mislabelling can't be used to dodge the check
        for cat in {p.category, p.predicted_category}:
            _, ids = self.img_idx[cat].search(p.img_embs, s.top_k)
            for pid in self.db.products_for_images(int(i) for i in ids.ravel() if i >= 0).values():
                cand.setdefault(pid, None)
            _, tids = self.txt_idx[cat].search(p.txt_emb, s.top_k)
            for pid in tids.ravel():
                if pid >= 0:
                    cand.setdefault(int(pid), None)

        # Photo fingerprint: across ALL categories
        for _, pid, h in self._hashes:
            d = min(int(h - q) for q in p.hashes)
            if d <= s.phash_max_distance:
                prev = cand.get(pid)
                cand[pid] = d if prev is None else min(prev, d)

        matches = []
        for prod in self.db.get_products(cand.keys()):
            if not s.compare_same_vendor and prod["vendor_id"] == vendor_id:
                continue
            m = self._score(p, price, prod, cand[prod["id"]])
            if m.level != "LOW":
                matches.append(m)
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[: s.max_matches]

    def _score(self, p: _Prepared, price, prod, phash_d) -> Match:
        s = self.s
        cat = prod["category"]

        stored = self.img_idx[cat].get(self.db.image_ids_for_product(prod["id"]))
        img_sim = float((p.img_embs @ stored.T).max()) if len(stored) else 0.0
        tvec = self.txt_idx[cat].get([prod["id"]])
        txt_sim = float(tvec[0] @ p.txt_emb) if len(tvec) else 0.0
        img_sim, txt_sim = max(0.0, img_sim), max(0.0, txt_sim)
        price_sim = _price_similarity(price, prod["price"])

        weights = {"image": s.w_image, "text": s.w_text}
        values = {"image": img_sim, "text": txt_sim}
        if price_sim is not None:
            weights["price"], values["price"] = s.w_price, price_sim
        score = sum(weights[k] * values[k] for k in weights) / sum(weights.values())

        reasons = []
        if phash_d is not None:
            score = max(score, s.phash_floor_score)
            reasons.append(f"Same or near-identical product photo (fingerprint distance {phash_d})")
        if img_sim >= 0.93:
            reasons.append(f"Item looks visually identical ({img_sim:.0%})")
        elif img_sim >= 0.85:
            reasons.append(f"Item looks visually similar ({img_sim:.0%})")
        if txt_sim >= 0.85:
            reasons.append(f"Near-identical title/description ({txt_sim:.0%})")
        elif txt_sim >= 0.70:
            reasons.append(f"Similar title/description ({txt_sim:.0%})")
        if price_sim is not None and price_sim >= 0.9:
            reasons.append("Price within 10%")

        level = "HIGH" if score >= s.high_risk else "MEDIUM" if score >= s.medium_risk else "LOW"
        return Match(prod["id"], prod["vendor_id"], prod["title"], cat, round(score, 4), level,
                     round(img_sim, 4), round(txt_sim, 4),
                     None if price_sim is None else round(price_sim, 4), phash_d, reasons)

    def _result(self, p: _Prepared, matches) -> CheckResult:
        level = matches[0].level if matches else "LOW"
        return CheckResult(p.category, p.category_confidence, p.category_warning,
                           level, level != "LOW", ACTIONS[level], matches)
