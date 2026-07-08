import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_qaq_batching_policy import (
    load_policy_batches,
    load_prompt_map,
    validate_prompts_for_batches,
)


def test_load_prompt_map_reads_jsonl_prompts(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        json.dumps({"request_id": "a", "prompt": "first"}) + "\n"
        + json.dumps({"request_id": "b", "text": "second"}) + "\n"
    )

    assert load_prompt_map(prompt_file) == {"a": "first", "b": "second"}


def test_load_prompt_map_rejects_duplicate_request_ids(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        json.dumps({"request_id": "a", "prompt": "first"}) + "\n"
        + json.dumps({"request_id": "a", "prompt": "second"}) + "\n"
    )

    with pytest.raises(ValueError, match="Duplicate request_id"):
        load_prompt_map(prompt_file)


def test_load_policy_batches_selects_policy_and_respects_limit(tmp_path):
    simulation_json = tmp_path / "simulation.json"
    simulation_json.write_text(json.dumps({
        "policies": {
            "scalar_budget_batching": {
                "batches": [
                    {"batch_id": "b0", "request_ids": ["a"]},
                    {"batch_id": "b1", "request_ids": ["b"]},
                ]
            }
        }
    }))

    _, batches = load_policy_batches(simulation_json, "scalar_budget_batching", max_batches=1)

    assert batches == [{"batch_id": "b0", "request_ids": ["a"]}]


def test_validate_prompts_for_batches_reports_missing_ids():
    batches = [{"batch_id": "b0", "request_ids": ["a", "missing"]}]

    with pytest.raises(ValueError, match="missing 1 request IDs"):
        validate_prompts_for_batches({"a": "prompt"}, batches)
