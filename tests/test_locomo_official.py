from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from scholar_mind.eval.locomo import load_official_locomo, run_official_locomo
from scholar_mind.main import cli_app


class _FakeMemoryManager:
    def __init__(self):
        self.logged_rounds: list[dict] = []
        self.extracted_users: list[str] = []
        self.llm = self

    def log_round(self, *, user_id, session_id, round_index, messages, explicit_memories=None):
        self.logged_rounds.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "round_index": round_index,
                "content": messages[0].content,
                "explicit_memories": explicit_memories,
            }
        )

    def extract_pending_memories(self, user_id: str | None = None):
        self.extracted_users.append(user_id or "")
        return 1

    def get_context_sync(self, *, user_id, current_query):
        return f"- remembered context for {user_id}: {current_query}", 1

    def invoke(self, _messages):
        return AIMessage(content="official answer prediction")


def _write_official_locomo_sample(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "conversation": {
                        "speaker_a": "Caroline",
                        "speaker_b": "Melanie",
                        "session_1_date_time": "2023-05-07",
                        "session_1": [
                            {
                                "dia_id": "D1:1",
                                "speaker": "Caroline",
                                "text": "I went to a support group today.",
                            },
                            {
                                "dia_id": "D1:2",
                                "speaker": "Melanie",
                                "text": "I painted a sunrise.",
                                "blip_caption": "a sunrise painting",
                            },
                        ],
                    },
                    "qa": [
                        {
                            "question": "When did Caroline go to the support group?",
                            "answer": "7 May 2023",
                            "evidence": ["D1:1"],
                            "category": 2,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_official_locomo_preserves_conversation_and_qa(tmp_path):
    data_path = tmp_path / "locomo10.json"
    _write_official_locomo_sample(data_path)

    samples = load_official_locomo(data_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_1"
    assert samples[0].speaker_a == "Caroline"
    assert samples[0].speaker_b == "Melanie"
    assert [turn.dialog_id for turn in samples[0].turns] == ["D1:1", "D1:2"]
    assert samples[0].turns[1].image_caption == "a sunrise painting"
    assert samples[0].questions[0].question == "When did Caroline go to the support group?"
    assert samples[0].questions[0].answer == "7 May 2023"
    assert samples[0].questions[0].evidence == ["D1:1"]
    assert samples[0].questions[0].category == 2


def test_run_official_locomo_writes_official_prediction_shape(tmp_path):
    data_path = tmp_path / "locomo10.json"
    out_path = tmp_path / "predictions.json"
    _write_official_locomo_sample(data_path)
    memory_manager = _FakeMemoryManager()

    result = run_official_locomo(
        data_file=data_path,
        out_file=out_path,
        memory_manager=memory_manager,
        model_key="scholarmind_memory",
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    qa = payload[0]["qa"][0]
    assert result == {
        "sample_count": 1,
        "question_count": 1,
        "out_file": str(out_path),
        "model_key": "scholarmind_memory",
    }
    assert payload[0]["sample_id"] == "sample_1"
    assert qa["question"] == "When did Caroline go to the support group?"
    assert qa["answer"] == "7 May 2023"
    assert qa["evidence"] == ["D1:1"]
    assert qa["category"] == 2
    assert qa["scholarmind_memory_prediction"] == "official answer prediction"
    assert qa["scholarmind_memory_memory_context"].startswith("- remembered context")
    assert qa["scholarmind_memory_memory_hit_count"] == 1
    assert len(payload[0]["qa"]) == 1
    assert memory_manager.logged_rounds[0]["content"].startswith(
        "[2023-05-07] Caroline (D1:1):"
    )
    assert memory_manager.extracted_users == ["locomo:sample_1"]


def test_locomo_cli_runs_official_loader_without_business_adaptation(
    tmp_path,
    monkeypatch,
):
    data_path = tmp_path / "locomo10.json"
    out_path = tmp_path / "predictions.json"
    _write_official_locomo_sample(data_path)
    memory_manager = _FakeMemoryManager()
    monkeypatch.setattr(
        "scholar_mind.main.get_container",
        lambda: SimpleNamespace(memory_manager=memory_manager),
    )

    result = CliRunner().invoke(
        cli_app,
        [
            "eval",
            "locomo",
            "--data-file",
            str(data_path),
            "--out-file",
            str(out_path),
            "--model-key",
            "scholarmind_memory",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))[0]["qa"][0][
        "scholarmind_memory_prediction"
    ] == "official answer prediction"
