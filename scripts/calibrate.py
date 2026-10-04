"""
Tune thresholds from moderator feedback.

After moderators mark alerts 'confirmed' (real copy) or 'dismissed' (false alarm)
via PATCH /alerts/{id}, run:

  python -m scripts.calibrate --target-precision 0.9

It prints precision per threshold and suggests HIGH_RISK / MEDIUM_RISK values.
"""
import argparse
from pathlib import Path

import numpy as np

from app.db import Database
from config import get_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    db = Database(Path(get_settings().data_dir) / "catalog.db")
    rows = [r for r in db.list_alerts(status=None, limit=10**9) if r["status"] in ("confirmed", "dismissed")]
    if len(rows) < 20:
        print(f"Only {len(rows)} reviewed alerts. Review at least ~20 (ideally 100+) before calibrating.")
        return
    scores = np.array([r["score"] for r in rows])
    y = np.array([r["status"] == "confirmed" for r in rows])
    print(f"{len(rows)} reviewed alerts, {y.sum()} confirmed\n")
    print("threshold  flagged  precision  recall")
    best = None
    for t in np.arange(0.75, 0.991, 0.01):
        sel = scores >= t
        if not sel.any():
            continue
        prec, rec = y[sel].mean(), y[sel].sum() / max(1, y.sum())
        print(f"  {t:.2f}     {sel.sum():5d}     {prec:.2f}     {rec:.2f}")
        if best is None and prec >= args.target_precision:
            best = t
    if best is not None:
        print(f"\nSuggested HIGH_RISK={best:.2f}  MEDIUM_RISK={max(0.75, best - 0.06):.2f}")
    else:
        print("\nTarget precision not reached at any threshold; raise thresholds or review more alerts.")
    print("Note: only scores above the current MEDIUM_RISK were ever stored, so lower values can't be evaluated.")


if __name__ == "__main__":
    main()
