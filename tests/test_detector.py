"""Logic tests with a tiny fake embedder (no model download needed).  Run: python -m pytest -q"""
import hashlib

import numpy as np
import pytest
from PIL import Image

from app.detector import SimilarityDetector
from config import Settings


class FakeEmbedder:
    image_dim, text_dim = 64, 32

    def embed_images(self, imgs):
        out = []
        for im in imgs:
            v = np.asarray(im.convert("L").resize((8, 8)), "float32").ravel() - 127.5
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out).astype("float32")

    def embed_text(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.text_dim, "float32")
            for w in t.lower().replace(".", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.text_dim] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)

    def classify(self, embs):
        return "bag", 0.99


def make_image(seed, size=256):
    rng = np.random.default_rng(seed)
    blocks = (rng.random((8, 8, 3)) * 255).astype("uint8")
    return Image.fromarray(blocks).resize((size, size), Image.NEAREST)


@pytest.fixture
def det(tmp_path):
    return SimilarityDetector(Settings(data_dir=str(tmp_path)), embedder=FakeEmbedder())


def test_copied_photo_by_other_vendor_is_high_risk(det):
    det.register("vendorA", "Black leather tote bag", price=25000, images=[make_image(1)])
    copied = make_image(1).resize((230, 230))  # re-saved / resized copy
    _, res = det.register("vendorB", "Leather tote bag black", price=24000, images=[copied])
    assert res.alarm and res.risk_level == "HIGH"
    assert res.matches[0].vendor_id == "vendorA"
    assert res.matches[0].photo_hash_distance is not None
    assert res.recommended_action == "hold_for_review"


def test_same_vendor_reupload_not_flagged(det):
    det.register("vendorA", "Black leather tote bag", images=[make_image(1)])
    _, res = det.register("vendorA", "Black leather tote bag", images=[make_image(1)])
    assert not res.alarm


def test_different_product_not_flagged(det):
    det.register("vendorA", "Black leather tote bag", images=[make_image(1)])
    _, res = det.register("vendorC", "Red running sneakers size 42", images=[make_image(99)])
    assert res.risk_level == "LOW" and res.matches == []


def test_check_does_not_save_and_removal_works(det):
    pid, _ = det.register("vendorA", "Black leather tote bag", images=[make_image(1)])
    assert det.check("vendorB", "tote", images=[make_image(1)]).alarm
    assert det.db.count_products() == 1
    assert det.remove_product(pid)
    assert not det.check("vendorB", "tote", images=[make_image(1)]).alarm


def test_alerts_persist_and_reload(det, tmp_path):
    det.register("vendorA", "Black leather tote bag", images=[make_image(1)])
    det.register("vendorB", "Black leather tote bag", images=[make_image(1)])
    assert len(det.db.list_alerts(status="open")) == 1
    det2 = SimilarityDetector(Settings(data_dir=str(tmp_path)), embedder=FakeEmbedder())
    assert det2.check("vendorC", "bag", images=[make_image(1)]).alarm
