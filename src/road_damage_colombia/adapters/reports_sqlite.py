"""Damage reports with their location, so the map can show where to send a crew first."""
import json
import sqlite3
import time
from pathlib import Path


class Reports:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""create table if not exists reports (id integer primary key, created real, lat real,
                           lng real, level text, score real, detections text, treatments text)""")

    def add(self, lat, lng, result: dict) -> int:
        cur = self.db.execute(
            "insert into reports (created, lat, lng, level, score, detections, treatments) values (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), lat, lng, result["level"], result["score"], json.dumps(result["detections"]),
             json.dumps(result["treatments"])))
        self.db.commit()
        return cur.lastrowid

    def all(self) -> list[dict]:
        rows = self.db.execute("select id, created, lat, lng, level, score, detections, treatments from reports "
                               "order by score desc").fetchall()
        return [{"id": r[0], "created": r[1], "lat": r[2], "lng": r[3], "level": r[4], "score": r[5],
                 "detections": json.loads(r[6]), "treatments": json.loads(r[7])} for r in rows]
