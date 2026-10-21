import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("quality_gate", Path(__file__).parents[1] / "scripts" / "quality_gate.py")


def test_gate_reports_every_limit_it_breaks():
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    limits = {"map50_japan": {"min": 0.55}, "median_latency_ms": {"max": 30}}
    assert gate.check({"map50_japan": 0.60, "median_latency_ms": 12}, limits) == []
    failures = gate.check({"map50_japan": 0.40, "median_latency_ms": 45}, limits)
    assert len(failures) == 2 and "below 0.55" in failures[0] and "above 30" in failures[1]
