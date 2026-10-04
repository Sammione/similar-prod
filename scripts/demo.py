"""
Quick local demo without the API:

  python -m scripts.demo vendorA path/to/bag1.jpg "Black leather tote bag" \
                         vendorB path/to/bag2.jpg "Leather tote black"
"""
import json
import sys

from app.detector import SimilarityDetector
from config import get_settings


def main():
    a = sys.argv[1:]
    if len(a) != 6:
        print(__doc__)
        sys.exit(1)
    det = SimilarityDetector(get_settings())
    pid1, r1 = det.register(a[0], a[2], images=[a[1]])
    print(f"Registered product {pid1} for {a[0]} (category: {r1.category})")
    pid2, r2 = det.register(a[3], a[5], images=[a[4]])
    print(f"Registered product {pid2} for {a[3]}")
    print(json.dumps(r2.to_dict(), indent=2))


if __name__ == "__main__":
    main()
