import copy

import pytest
import yaml

from fmsr.runinfo import RunConfigMismatch, write_run_config


def _cfg():
    return {
        "variant": "V",
        "prompt_id": "i1",
        "prompt_text": "remove the shadow",
        "resolution": [256, 256],
        "sampling_params": {"num_inference_steps": 50},
        "versions": {"torch": "2.11.0", "cudnn": 1},
        "repo_git_sha": "aaa",
        "created_at": "t0",
    }


def test_no_clobber_and_informational_fields_ignored(tmp_path):
    path = tmp_path / "run_config.yaml"
    assert write_run_config(path, _cfg()) == "created"
    again = copy.deepcopy(_cfg())
    again.update(repo_git_sha="bbb", created_at="t1")
    again["versions"]["cudnn"] = 2
    assert write_run_config(path, again) == "matched"
    assert yaml.safe_load(path.read_text())["repo_git_sha"] == "aaa"
    assert [p.name for p in tmp_path.iterdir()] == ["run_config.yaml"]


def test_repro_key_mismatch_aborts(tmp_path):
    path = tmp_path / "run_config.yaml"
    write_run_config(path, _cfg())
    changed = _cfg()
    changed["prompt_text"] = "remove shadows"
    with pytest.raises(RunConfigMismatch, match="prompt_text"):
        write_run_config(path, changed)
    changed = _cfg()
    changed["versions"]["torch"] = "2.12.0"
    with pytest.raises(RunConfigMismatch, match="versions.torch"):
        write_run_config(path, changed)
