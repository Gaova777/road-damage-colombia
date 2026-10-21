import io

from fastapi.testclient import TestClient
from PIL import Image

from road_damage_colombia.adapters.reports_sqlite import Reports
from road_damage_colombia.domain.damage import Detection
from road_damage_colombia.entrypoints import api


class FakeDetector:
    name = "fake"

    def detect(self, image):
        return [Detection("D40", 0.9, (0.4, 0.6, 0.6, 0.8))]


def photo():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def test_detect_saves_a_located_report_and_counts_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "detector", lambda: FakeDetector())
    store = Reports(tmp_path / "r.sqlite")
    monkeypatch.setattr(api, "reports", lambda: store)
    c = TestClient(api.app)
    r = c.post("/detect", files={"photo": ("a.jpg", photo(), "image/jpeg")}, data={"lat": "4.81", "lng": "-75.69"}).json()
    assert r["level"] == "urgent" and r["detections"][0]["treatment"] == "pothole repair" and r["id"] == 1
    assert c.get("/reports").json()[0]["lat"] == 4.81
    assert c.post("/detect", files={"photo": ("a.txt", b"nope", "text/plain")}).status_code == 400
    assert 'rdc_detections_total{cls="D40"}' in c.get("/metrics").text
