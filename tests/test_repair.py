"""Smoke tests for repair utilities."""

from repair.geometric_confidence import GeometricConfidence
from repair.monitors import Monitor


def test_geometric_confidence_combined():
    gc = GeometricConfidence()
    result = gc.combined(0.8, 0.6, 0.7)
    expected = 0.2 * 0.8 + 0.5 * 0.6 + 0.3 * 0.7
    assert abs(result - expected) < 1e-6


def test_geometric_confidence_normalize():
    gc = GeometricConfidence()
    assert abs(gc.normalize(0.5) - 0.5) < 1e-6
    assert gc.normalize(-1.0) == 0.0
    assert gc.normalize(2.0) == 1.0


def test_monitor_log_and_summary(tmp_path):
    mon = Monitor({'log_interval': 5, 'output_dir': str(tmp_path)})
    for i in range(20):
        mon.log(i, {
            'confidence': 0.5 + i * 0.01,
            'dist_to_ref': 1.0 - i * 0.01,
            'repair_cost_seconds': 0.001,
            'repair_triggered': i % 3 == 0,
        })
    mon.save()
    summary = mon.summary()

    assert summary['total_steps'] == 20
    assert 0.0 <= summary['repair_rate'] <= 1.0
    assert (tmp_path / 'metrics.csv').exists()
