import json
import os

import numpy as np
import pytest

from acquisition import validate_dataset as vd, validate_license as vl, inspect_survey

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "output", "diagnostics", "dataset_manifest.json")
needs_data = pytest.mark.skipif(not os.path.exists(MAN), reason="run scripts/inspect_dataset.py first")


def test_license_validator_flags_unknown_licence():
    ok = vl.validate({"name": "x", "license": "public domain", "license_url": "u", "provider": "p", "restrictions": "none"})
    assert ok["valid"] and not ok["needs_review"]
    bad = vl.validate({"name": "x", "license": "unknown", "license_url": "", "provider": "p", "restrictions": "?"})
    assert not bad["valid"] and bad["needs_review"]


def test_manifest_checker_detects_gaps():
    r = vd.check_manifest({"dataset_name": "a"})
    assert not r["complete"] and "file_hashes" in r["missing"]


@needs_data
def test_manifest_complete_and_hashes_match():
    import hashlib
    m = json.load(open(MAN))
    assert vd.check_manifest(m)["complete"]
    assert m["synthetic_flags"] and m["missing_requirements"]               # honest about what is simulated / absent
    for f, h in m["file_hashes"].items():
        p = os.path.join(ROOT, f)
        assert os.path.exists(p)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h


@needs_data
def test_real_files_exist_and_validate():
    m = json.load(open(MAN))
    for f in m["downloaded_files"]:
        p = os.path.join(ROOT, f)
        assert os.path.getsize(p) > 100000
        r = vd.check_geotiff(p)
        assert r["readable"] and r["has_data"] and r["float_elevation"] and r["crs"] == "EPSG:4326"


@needs_data
def test_survey_inspection_numbers():
    info = inspect_survey.inspect(os.path.join(ROOT, "data", "bathymetry", "monterey_survey_native.tif"))
    assert info["valid_fraction"] > 0.99 and info["elevation_max_m"] < 0 < 1
    assert 3.0 < info["cell_m_approx"][0] < 4.0


def test_simulation_is_labelled_simulated():
    p = os.path.join(ROOT, "data", "simulation", "sim_meta.json")
    if not os.path.exists(p):
        pytest.skip("simulation not generated")
    m = json.load(open(p))
    assert m["simulated"] is True and "SIMULATED" in m["label"]
    assert all(o["synthetic"] for o in m["objects"])
