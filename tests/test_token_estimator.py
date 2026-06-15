from __future__ import annotations

import pytest

from scholar_mind.utils.token_estimator import estimate_text_tokens


def test_estimate_text_tokens_uses_model_family_encoding(monkeypatch):
    calls: list[str] = []

    class _Encoding:
        def encode(self, text: str) -> list[int]:
            assert text == "hello"
            return [1, 2, 3, 4]

    def fake_get_encoding(name: str):
        calls.append(name)
        if name == "o200k_base":
            return _Encoding()
        raise AssertionError(name)

    monkeypatch.setattr("scholar_mind.utils.token_estimator._get_encoding", fake_get_encoding)

    assert estimate_text_tokens("hello", model_name="openai/gpt-5.4") == 4
    assert calls == ["o200k_base"]


def test_estimate_text_tokens_raises_when_model_encoding_unavailable(monkeypatch):
    monkeypatch.setattr(
        "scholar_mind.utils.token_estimator._get_encoding",
        lambda _name: (_ for _ in ()).throw(RuntimeError("encoding unavailable")),
    )

    with pytest.raises(RuntimeError, match="encoding unavailable"):
        estimate_text_tokens("用户偏好中文回答", model_name="openai/gpt-5.4")
