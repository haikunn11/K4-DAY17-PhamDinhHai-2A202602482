from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates
from model_provider import normalize_provider


def make_config(tmp_path: Path):
    base = load_config(tmp_path)
    return replace(
        base,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=90,
        compact_keep_messages=4,
    )


def test_provider_normalization() -> None:
    assert normalize_provider(" OpenAI ") == "openai"
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("google-genai") == "gemini"
    assert normalize_provider("openai-compatible") == "custom"
    with pytest.raises(ValueError, match="Unsupported provider"):
        normalize_provider("unknown")


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("dungct") == 0
    path = store.write_text("dungct", "# User Profile\n\n- location: Đà Nẵng")
    assert path.exists()
    assert "Đà Nẵng" in store.read_text("dungct")
    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct")
    assert store.file_size("dungct") > 0
    assert store.path_for("../../unsafe").is_relative_to((tmp_path / "profiles").resolve())


def test_profile_extraction_handles_corrections_and_noise() -> None:
    correction = extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer. "
        "Mình vẫn ở Huế và muốn trả lời ngắn gọn."
    )
    assert correction["profession"] == "MLOps engineer"
    assert correction["location"] == "Huế"
    noise = extract_profile_updates(
        "Mình đùa là chuyển sang product manager. Hà Nội chỉ là nơi mình bay ra họp hai ngày; "
        "nghề nghiệp hiện tại vẫn là MLOps engineer."
    )
    assert noise["profession"] == "MLOps engineer"
    assert "location" not in noise
    assert "profession" not in extract_profile_updates("Mình làm việc từ xa với team cũ.")
    assert "profession" not in extract_profile_updates("Nếu compact tốt thì agent đang làm đúng việc của nó.")


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(10):
        agent.reply("user", "long-thread", f"Lượt {index}: " + "ngữ cảnh dài " * 30)
    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    statement = "Mình tên là DũngCT, hiện tại mình ở Huế và đang làm MLOps engineer."
    baseline.reply("dungct", "thread-1", statement)
    advanced.reply("dungct", "thread-1", statement)
    question = "Sang thread mới, nhắc lại tên, nơi ở và nghề nghiệp hiện tại của mình?"
    baseline_answer = baseline.reply("dungct", "thread-2", question)["response"]
    advanced_answer = advanced.reply("dungct", "thread-2", question)["response"]
    assert "DũngCT" not in baseline_answer
    assert all(fact in advanced_answer for fact in ("DũngCT", "Huế", "MLOps engineer"))


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("user", "one", "Mình ở Đà Nẵng và đang làm backend engineer.")
    agent.reply("user", "two", "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("user", "three", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    answer = agent.reply("user", "recall", "Hiện tại mình ở đâu và làm nghề gì?")["response"]
    assert "Huế" in answer and "MLOps engineer" in answer
    profile = agent.profile_store.read_text("user")
    assert "- location: Đà Nẵng" not in profile
    assert "- profession: backend engineer" not in profile


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Lượt {index}: " + "đây là một đoạn hội thoại dài để đo chi phí prompt " * 20
        baseline.reply("user", "long", message)
        advanced.reply("user", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")


@pytest.mark.parametrize("dataset_name", ["conversations.json", "advanced_long_context.json"])
def test_advanced_recalls_all_dataset_facts(tmp_path: Path, dataset_name: str) -> None:
    from benchmark import recall_points

    root = Path(__file__).resolve().parent.parent
    conversations = json.loads((root / "data" / dataset_name).read_text(encoding="utf-8"))
    config = replace(make_config(tmp_path), compact_threshold_tokens=1_200, compact_keep_messages=6)
    agent = AdvancedAgent(config, force_offline=True)
    for conversation in conversations:
        user_id = conversation["user_id"]
        for turn in conversation["turns"]:
            agent.reply(user_id, conversation["id"], turn)
        for index, recall in enumerate(conversation["recall_questions"]):
            answer = agent.reply(user_id, f"{conversation['id']}-recall-{index}", recall["question"])["response"]
            assert recall_points(answer, recall["expected_contains"]) == 1.0
