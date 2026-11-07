import json

import pytest

from benchmarks.drvoi_v12.metrics import (
    METRIC_SCHEMA_VERSION,
    REQUIRED_UPDATE_METRICS,
    append_update_metrics,
    validate_update_metrics,
)


def complete_metrics():
    return {name: float(index) for index, name in enumerate(
        REQUIRED_UPDATE_METRICS)}


def test_metric_contract_rejects_missing_and_nonfinite_values():
    metrics = complete_metrics()
    metrics.pop("hits")
    with pytest.raises(ValueError, match="missing"):
        validate_update_metrics(metrics)
    metrics = complete_metrics()
    metrics["router_approx_kl"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        validate_update_metrics(metrics)


def test_metric_jsonl_saves_every_required_group(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = complete_metrics()
    append_update_metrics(path, metrics)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["metric_schema"] == METRIC_SCHEMA_VERSION
    assert all(key in record for key in REQUIRED_UPDATE_METRICS)
