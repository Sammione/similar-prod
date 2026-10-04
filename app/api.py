"""FastAPI service. Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from config import get_settings
from .detector import SimilarityDetector, load_image

detector: Optional[SimilarityDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = SimilarityDetector(get_settings())   # loads pretrained models once
    yield
    detector.save()


app = FastAPI(title="Vendor Product Similarity Guard", version="1.0", lifespan=lifespan)


def _read_images(files: List[UploadFile]):
    if not files:
        raise HTTPException(400, "At least one image is required")
    out = []
    for f in files:
        try:
            out.append(load_image(f.file.read()))
        except Exception:
            raise HTTPException(400, f"Could not read image '{f.filename}'")
    return out


@app.get("/health")
def health():
    return {"status": "ok", **detector.stats()}


@app.post("/products")
def upload_product(
    vendor_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    price: Optional[float] = Form(None),
    category: Optional[str] = Form(None, description="bag | shoe | clothing (auto-detected if empty)"),
    external_id: Optional[str] = Form(None, description="Your platform's product ID"),
    images: List[UploadFile] = File(...),
):
    """Call this when a vendor uploads a product. Scores it, saves it, and records alerts."""
    try:
        pid, result = detector.register(vendor_id, title, description, price, _read_images(images),
                                        category, external_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"product_id": pid, **result.to_dict()}


@app.post("/products/check")
def check_product(
    vendor_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    price: Optional[float] = Form(None),
    category: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
):
    """Dry run: score a product without saving it (e.g. live warning on the vendor upload form)."""
    try:
        result = detector.check(vendor_id, title, description, price, _read_images(images), category)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result.to_dict()


@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    if not detector.remove_product(product_id):
        raise HTTPException(404, "Product not found")
    return {"deleted": product_id}


@app.get("/alerts")
def list_alerts(status: Optional[str] = Query("open"), level: Optional[str] = None, limit: int = 100):
    return detector.db.list_alerts(status=status, level=level, limit=limit)


class AlertUpdate(BaseModel):
    status: Literal["open", "confirmed", "dismissed"]
    note: Optional[str] = None


@app.patch("/alerts/{alert_id}")
def review_alert(alert_id: int, body: AlertUpdate):
    """Moderator feedback: 'confirmed' = real copy, 'dismissed' = false alarm. Used by scripts/calibrate.py."""
    if not detector.db.update_alert(alert_id, body.status, body.note):
        raise HTTPException(404, "Alert not found")
    return {"id": alert_id, "status": body.status}
