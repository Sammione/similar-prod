# Vendor Product Similarity Guard

Flags when a vendor uploads a product (bags, shoes, clothing) that is similar to a product
already listed by a **different** vendor, and raises a risk alarm.

## How it works

When a product is uploaded, three signals are computed with pretrained models (no training needed):

| Signal | Model | Catches |
|---|---|---|
| Image similarity | **FashionCLIP** (`patrickjohncyh/fashion-clip`), CLIP fine-tuned on fashion products | Same item photographed differently, different angle/background |
| Text similarity | **all-MiniLM-L6-v2** sentence embeddings of title + description | Copied or lightly reworded listings |
| Photo fingerprint | Perceptual hash (pHash) | Stolen photos that were re-saved, resized, recompressed or lightly edited |
| Price closeness | — (small weight) | Supporting signal |

FashionCLIP also auto-detects the category (bag / shoe / clothing) by zero-shot classification.
The search runs in both the declared and the detected category, so mislabelling a product doesn't let it dodge the check.

Products are stored in SQLite, vectors in FAISS (one index per category). Products from the
**same vendor are ignored** (set `COMPARE_SAME_VENDOR=true` to also catch self-duplicates).

### Risk levels

`score = 0.6·image + 0.3·text + 0.1·price` (weights renormalised if price is missing).
A photo-fingerprint hit lifts the score to at least 0.97.

| Level | Default score | `recommended_action` |
|---|---|---|
| HIGH | ≥ 0.90 | `hold_for_review` — keep it off the storefront until a moderator checks |
| MEDIUM | ≥ 0.82 | `publish_and_flag` — publish but put it in the review queue |
| LOW | below | `publish` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                   # logic tests (no model download)
uvicorn app.api:app --host 0.0.0.0 --port 8000        # first start downloads ~700 MB of models
```
Interactive docs: http://localhost:8000/docs

Docker: `docker build -t similarity-guard . && docker run -p 8000:8000 -v $(pwd)/data:/data similarity-guard`

## Seed your existing catalogue

```bash
# CSV columns: vendor_id,title,description,price,category,images,external_id   (images separated by ';')
python -m scripts.bulk_index products.csv --images-root ./product_images
python -m scripts.bulk_index products.csv --images-root ./product_images --check   # also flag existing duplicates
```

## API

**Upload (score + save + record alerts)** — call from your backend when a vendor uploads:
```bash
curl -X POST http://localhost:8000/products \
  -F vendor_id=vendor_42 -F title="Black leather tote bag" -F price=25000 \
  -F category=bag -F external_id=SKU-991 \
  -F images=@front.jpg -F images=@side.jpg
```
Response:
```json
{
  "product_id": 118,
  "category": "bag", "category_confidence": 0.97, "category_warning": null,
  "risk_level": "HIGH", "alarm": true, "recommended_action": "hold_for_review",
  "matches": [{
    "product_id": 37, "vendor_id": "vendor_7", "title": "Leather tote bag - black",
    "score": 0.97, "level": "HIGH",
    "image_similarity": 0.95, "text_similarity": 0.88, "price_similarity": 0.96,
    "photo_hash_distance": 2,
    "reasons": ["Same or near-identical product photo (fingerprint distance 2)",
                "Item looks visually identical (95%)", "Near-identical title/description (88%)",
                "Price within 10%"]
  }]
}
```

| Endpoint | Purpose |
|---|---|
| `POST /products` | Score, save, create alerts |
| `POST /products/check` | Dry run, nothing saved (e.g. live warning on the vendor's upload form) |
| `DELETE /products/{id}` | Remove a product from the index (product deleted or taken down) |
| `GET /alerts?status=open&level=HIGH` | Moderation queue |
| `PATCH /alerts/{id}` `{"status":"confirmed"\|"dismissed","note":"..."}` | Moderator feedback |
| `GET /health` | Status and counts |

## Tuning (important)

The default thresholds are a reasonable start, but similarity scores depend on your catalogue.
Generic items (plain white sneakers, black tote bags) genuinely look alike across vendors,
so expect some MEDIUM alerts that are honest coincidences. That's why MEDIUM → review, not block.

1. Run for a week or two and have moderators mark alerts `confirmed` or `dismissed`.
2. `python -m scripts.calibrate --target-precision 0.9` prints precision per threshold and suggests values.
3. Set them via env vars: `HIGH_RISK`, `MEDIUM_RISK`, `W_IMAGE`, `W_TEXT`, `W_PRICE`, `PHASH_MAX_DISTANCE`.

All settings are in `config.py` and can be overridden by environment variables
(e.g. `DEVICE=cuda`, `DATA_DIR=/data`, `IMAGE_MODEL=openai/clip-vit-base-patch32`).
If you change a model, delete the data directory and re-index (vector sizes change).

## Scaling notes

- Exact FAISS search is fine up to a few hundred thousand images. Beyond that, switch
  `IndexFlatIP` to `IndexHNSWFlat` in `app/vector_index.py`.
- The photo-fingerprint check is a linear scan in memory (fast to ~200k images); use a BK-tree beyond that.
- Run a single API worker (state is in-process). For multiple workers, move vectors to a vector DB
  (Qdrant / pgvector) and metadata to Postgres.
- GPU is optional; on CPU expect roughly 0.1–0.3 s per image.

## Project layout
```
config.py              settings (env-overridable)
app/embedder.py        FashionCLIP + MiniLM wrapper, zero-shot category
app/vector_index.py    FAISS wrapper
app/db.py              SQLite: products, images, alerts
app/detector.py        matching + risk scoring (core logic)
app/api.py             FastAPI service
scripts/bulk_index.py  seed existing catalogue from CSV
scripts/calibrate.py   tune thresholds from moderator feedback
scripts/demo.py        two-product command-line demo
tests/                 logic tests with a fake embedder
```
