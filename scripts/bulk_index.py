"""
Seed the catalogue with your existing products.

CSV columns: vendor_id,title,description,price,category,images,external_id
  images = image paths separated by ';' (relative to --images-root)

  python -m scripts.bulk_index products.csv --images-root ./product_images
  python -m scripts.bulk_index products.csv --check     # also raise alerts between existing products
"""
import argparse
import csv
from pathlib import Path

from app.detector import SimilarityDetector
from config import get_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--images-root", default=".")
    ap.add_argument("--check", action="store_true", help="create alerts for duplicates already in the catalogue")
    args = ap.parse_args()

    det = SimilarityDetector(get_settings())
    root = Path(args.images_root)
    n_ok = n_flag = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            paths = [root / p.strip() for p in (row.get("images") or "").split(";") if p.strip()]
            paths = [p for p in paths if p.exists()]
            if not paths:
                print(f"[row {i}] skipped: no readable images")
                continue
            try:
                pid, res = det.register(
                    vendor_id=row["vendor_id"], title=row.get("title", ""),
                    description=row.get("description", ""), price=row.get("price") or None,
                    images=paths, declared_category=row.get("category") or None,
                    external_id=row.get("external_id") or None,
                    create_alerts=args.check, save=False)
            except Exception as e:
                print(f"[row {i}] error: {e}")
                continue
            n_ok += 1
            if args.check and res.alarm:
                n_flag += 1
                top = res.matches[0]
                print(f"[row {i}] {res.risk_level}: '{row.get('title')}' ~ product {top.product_id} "
                      f"(vendor {top.vendor_id}, score {top.score})")
            if n_ok % 200 == 0:
                det.save()
                print(f"... {n_ok} indexed")
    det.save()
    print(f"Indexed {n_ok} products" + (f", {n_flag} flagged" if args.check else ""))


if __name__ == "__main__":
    main()
