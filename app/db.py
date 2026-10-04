"""SQLite storage for products, images (with perceptual hashes) and risk alerts."""
from __future__ import annotations

import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_id TEXT NOT NULL,
    external_id TEXT,
    title TEXT,
    description TEXT,
    price REAL,
    category TEXT,
    predicted_category TEXT,
    active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES products(id),
    path TEXT,
    phash TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_product ON images(product_id);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    matched_product_id INTEGER,
    vendor_id TEXT,
    matched_vendor_id TEXT,
    score REAL,
    level TEXT,
    details TEXT,
    status TEXT DEFAULT 'open',      -- open | confirmed | dismissed
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
"""


def _ph(n):
    return ",".join("?" * n)


class Database:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def _x(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cur.lastrowid, cur.rowcount

    # ---- products ----
    def insert_product(self, vendor_id, title, description, price, category, predicted_category, external_id=None):
        pid, _ = self._x(
            "INSERT INTO products(vendor_id, external_id, title, description, price, category, predicted_category)"
            " VALUES (?,?,?,?,?,?,?)",
            (vendor_id, external_id, title, description, price, category, predicted_category))
        return pid

    def get_product(self, pid):
        rows = self._q("SELECT * FROM products WHERE id=?", (pid,))
        return rows[0] if rows else None

    def get_products(self, ids):
        ids = list(ids)
        if not ids:
            return []
        return self._q(f"SELECT * FROM products WHERE active=1 AND id IN ({_ph(len(ids))})", ids)

    def deactivate_product(self, pid):
        self._x("UPDATE products SET active=0 WHERE id=?", (pid,))

    def count_products(self):
        return self._q("SELECT COUNT(*) AS n FROM products WHERE active=1")[0]["n"]

    # ---- images ----
    def insert_image(self, product_id, path, phash):
        iid, _ = self._x("INSERT INTO images(product_id, path, phash) VALUES (?,?,?)", (product_id, path, phash))
        return iid

    def image_ids_for_product(self, pid):
        return [r["id"] for r in self._q("SELECT id FROM images WHERE product_id=?", (pid,))]

    def products_for_images(self, image_ids):
        image_ids = list(image_ids)
        if not image_ids:
            return {}
        rows = self._q(f"SELECT id, product_id FROM images WHERE id IN ({_ph(len(image_ids))})", image_ids)
        return {r["id"]: r["product_id"] for r in rows}

    def all_hashes(self):
        return self._q("SELECT i.id, i.product_id, i.phash FROM images i "
                       "JOIN products p ON p.id=i.product_id WHERE p.active=1")

    # ---- alerts ----
    def insert_alert(self, product_id, vendor_id, match):
        aid, _ = self._x(
            "INSERT INTO alerts(product_id, matched_product_id, vendor_id, matched_vendor_id, score, level, details)"
            " VALUES (?,?,?,?,?,?,?)",
            (product_id, match["product_id"], vendor_id, match["vendor_id"], match["score"], match["level"],
             json.dumps(match)))
        return aid

    def list_alerts(self, status=None, level=None, limit=100):
        sql, params = "SELECT * FROM alerts WHERE 1=1", []
        if status:
            sql += " AND status=?"; params.append(status)
        if level:
            sql += " AND level=?"; params.append(level)
        sql += " ORDER BY id DESC LIMIT ?"; params.append(limit)
        rows = self._q(sql, params)
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
        return rows

    def update_alert(self, alert_id, status, note=None):
        _, n = self._x("UPDATE alerts SET status=?, note=? WHERE id=?", (status, note, alert_id))
        return n > 0
