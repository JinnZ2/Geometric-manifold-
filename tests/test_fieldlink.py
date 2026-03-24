"""Smoke tests for bidirectional fieldlink sync with Rosetta-Shape-Core."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# .fieldlink.json structure
# ---------------------------------------------------------------------------


def test_fieldlink_json_exists():
    assert (ROOT / ".fieldlink.json").exists()


def test_fieldlink_json_valid():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    fl = data["fieldlink"]
    assert fl["version"] == "2.0"
    assert "sources" in fl
    assert "exports" in fl


def test_fieldlink_has_rosetta_source():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    sources = data["fieldlink"]["sources"]
    names = [s["name"] for s in sources]
    assert "rosetta" in names


def test_fieldlink_exports_defined():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    exports = data["fieldlink"]["exports"]
    export_names = [e["name"] for e in exports]
    assert "manifold_invariants" in export_names
    assert "basin_topology" in export_names
    assert "sync_manifest" in export_names


# ---------------------------------------------------------------------------
# Identity & peers (v2 bidirectional)
# ---------------------------------------------------------------------------


def test_fieldlink_has_identity():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    identity = data["fieldlink"]["identity"]
    assert identity["name"] == "geometric-manifold"
    assert "repo" in identity
    assert identity["namespace"] == "BASIN"


def test_fieldlink_has_peers():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    peers = data["fieldlink"]["peers"]
    assert len(peers) > 0
    names = [p["name"] for p in peers]
    assert "rosetta" in names


def test_peer_is_bidirectional():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    rosetta = next(p for p in data["fieldlink"]["peers"] if p["name"] == "rosetta")
    assert rosetta["direction"] == "bidirectional"


def test_peer_has_inbound_and_outbound():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    rosetta = next(p for p in data["fieldlink"]["peers"] if p["name"] == "rosetta")
    assert "inbound" in rosetta
    assert "outbound" in rosetta
    assert "paths" in rosetta["inbound"]
    assert "paths" in rosetta["outbound"]
    assert "mount_root" in rosetta["inbound"]
    assert "mount_root" in rosetta["outbound"]


def test_peer_has_conflict_resolution():
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    rosetta = next(p for p in data["fieldlink"]["peers"] if p["name"] == "rosetta")
    sync = rosetta["sync"]
    assert "conflict_resolution" in sync
    assert "conflict_rules" in sync
    assert len(sync["conflict_rules"]) > 0


def test_conflict_rules_have_authority():
    """Every conflict rule specifies who owns the contested path."""
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    rosetta = next(p for p in data["fieldlink"]["peers"] if p["name"] == "rosetta")
    for rule in rosetta["sync"]["conflict_rules"]:
        assert "pattern" in rule
        assert "authority" in rule
        assert rule["authority"] in ("rosetta", "geometric-manifold")


def test_conflict_rules_cover_key_paths():
    """Conflict rules must cover shapes (rosetta-owned) and exports (self-owned)."""
    data = json.loads((ROOT / ".fieldlink.json").read_text())
    rosetta = next(p for p in data["fieldlink"]["peers"] if p["name"] == "rosetta")
    rules = rosetta["sync"]["conflict_rules"]
    patterns = [r["pattern"] for r in rules]
    authorities = {r["pattern"]: r["authority"] for r in rules}

    assert "shapes/**" in patterns
    assert authorities["shapes/**"] == "rosetta"
    assert "atlas/exports/**" in patterns
    assert authorities["atlas/exports/**"] == "geometric-manifold"


# ---------------------------------------------------------------------------
# Bridge definitions
# ---------------------------------------------------------------------------


def test_bridge_file_exists():
    assert (ROOT / "bridges" / "rosetta-bridge.json").exists()


def test_bridge_valid_json():
    data = json.loads((ROOT / "bridges" / "rosetta-bridge.json").read_text())
    assert "layer_shape_map" in data
    assert "confidence_aggregation" in data
    assert "id_namespace" in data


def test_bridge_covers_all_layers():
    data = json.loads((ROOT / "bridges" / "rosetta-bridge.json").read_text())
    layers = [entry["layer"] for entry in data["layer_shape_map"]]
    assert "data_manifold" in layers
    assert "parameter_manifold" in layers
    assert "policy_manifold" in layers


def test_bridge_shape_ids_valid():
    """All shape references use SHAPE.X format matching Rosetta convention."""
    data = json.loads((ROOT / "bridges" / "rosetta-bridge.json").read_text())
    for entry in data["layer_shape_map"]:
        assert entry["shape"].startswith("SHAPE."), f"Invalid shape ID: {entry['shape']}"
    assert data["confidence_aggregation"]["shape"].startswith("SHAPE.")


def test_bridge_namespace():
    data = json.loads((ROOT / "bridges" / "rosetta-bridge.json").read_text())
    ns = data["id_namespace"]
    assert ns["prefix"] == "BASIN"
    for entity in ns["entities"]:
        assert entity.startswith("BASIN."), f"Entity {entity} missing BASIN prefix"


# ---------------------------------------------------------------------------
# Export generation
# ---------------------------------------------------------------------------


def test_export_script_runs(tmp_path):
    """Export script generates all three JSON files without error."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Export failed: {result.stderr}"
    assert (tmp_path / "manifold_invariants.json").exists()
    assert (tmp_path / "basin_topology.json").exists()
    assert (tmp_path / "sync_manifest.json").exists()


def test_invariants_export_structure(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "manifold_invariants.json").read_text())
    assert "$schema" in data
    assert "layers" in data
    layers = data["layers"]
    assert "parameter_manifold" in layers
    assert "policy_manifold" in layers
    assert "data_manifold" in layers
    assert "confidence_aggregation" in layers

    # Each layer should have invariants list
    for name, layer in layers.items():
        assert "invariants" in layer, f"{name} missing invariants"
        assert len(layer["invariants"]) > 0, f"{name} has empty invariants"
        assert "rosetta_shape" in layer, f"{name} missing rosetta_shape"


def test_topology_export_structure(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "basin_topology.json").read_text())
    assert "$schema" in data
    assert "trust_region" in data
    assert data["trust_region"]["type"] == "l2_ball"
    assert "curvature" in data
    assert "confidence_geometry" in data
    assert "loss_landscape" in data
    assert data["loss_landscape"]["type"] == "saddle_point"
    assert "pipeline" in data
    assert "rosetta_bridge" in data


def test_topology_shape_assignments(tmp_path):
    """All five Platonic solids are assigned to framework components."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "basin_topology.json").read_text())
    assignments = data["rosetta_bridge"]["shape_assignments"]
    shapes_used = set(assignments.values())
    assert "SHAPE.ICOSA" in shapes_used
    assert "SHAPE.DODECA" in shapes_used
    assert "SHAPE.OCTA" in shapes_used
    assert "SHAPE.TETRA" in shapes_used
    assert "SHAPE.CUBE" in shapes_used


# ---------------------------------------------------------------------------
# Sync manifest (v2 bidirectional)
# ---------------------------------------------------------------------------


def test_sync_manifest_structure(tmp_path):
    """Sync manifest has required fields for peer discovery."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "sync_manifest.json").read_text())
    assert data["$schema"] == "urn:fieldlink:sync-manifest:v2"
    assert "identity" in data
    assert "exports" in data
    assert "peers" in data
    assert "capabilities" in data


def test_sync_manifest_identity(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "sync_manifest.json").read_text())
    identity = data["identity"]
    assert identity["name"] == "geometric-manifold"
    assert identity["namespace"] == "BASIN"
    assert identity["share_ok"] is True


def test_sync_manifest_peers(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "sync_manifest.json").read_text())
    peers = data["peers"]
    assert len(peers) > 0
    rosetta = next(p for p in peers if p["name"] == "rosetta")
    assert rosetta["direction"] == "bidirectional"
    assert len(rosetta["conflict_rules"]) > 0


def test_sync_manifest_capabilities(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "sync_manifest.json").read_text())
    caps = data["capabilities"]
    assert caps["pull"] is True
    assert caps["push"] is True
    assert caps["conflict_resolution"] is True
    assert caps["hash_algorithm"] == "sha256"


def test_sync_manifest_lists_all_exports(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fieldlink_export.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "sync_manifest.json").read_text())
    export_names = [e["name"] for e in data["exports"]]
    assert "manifold_invariants" in export_names
    assert "basin_topology" in export_names
    assert "sync_manifest" in export_names
