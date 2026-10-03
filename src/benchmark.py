from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Benchmark dataset must contain a JSON list: {path}")
    required = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(data):
        if not isinstance(conversation, dict) or not required.issubset(conversation):
            raise ValueError(f"Conversation #{index} is missing required fields")
        if not isinstance(conversation["turns"], list) or not isinstance(conversation["recall_questions"], list):
            raise ValueError(f"Conversation {conversation['id']!r} has invalid turns or recall_questions")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    folded = answer.casefold()
    hits = sum(1 for fact in expected if fact.casefold() in folded)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concise = 1.0 if len(answer) <= 700 else 0.5
    structured = 1.0 if ("\n- " in answer or answer.startswith("- ")) else 0.5
    return round(0.7 * recall + 0.15 * concise + 0.15 * structured, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    del config
    thread_ids: set[str] = set()
    user_ids = {str(conversation["user_id"]) for conversation in conversations}
    before_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in user_ids
    }
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        conversation_id = str(conversation["id"])
        thread_id = f"{agent_name}-{conversation_id}"
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, item in enumerate(conversation["recall_questions"]):
            recall_thread = f"{agent_name}-{conversation_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(item["question"]))
            answer = str(result["response"])
            expected = [str(value) for value in item.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in user_ids
    }
    memory_growth = sum(max(0, after_sizes[user_id] - before_sizes[user_id]) for user_id in user_ids)
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=round(average_recall, 3),
        response_quality=round(average_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in values)) for i in range(len(headers))]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in values]
    return "\n".join([header, separator, *body])


def _run_suite(title: str, conversations: list[dict[str, Any]], base_config, state_root: Path) -> None:
    rows: list[BenchmarkRow] = []
    for agent_name, agent_type in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        agent_config = replace(base_config, state_dir=state_root / title.lower().replace(" ", "-") / agent_name.lower())
        agent_config.state_dir.mkdir(parents=True, exist_ok=True)
        agent = agent_type(agent_config, force_offline=True)
        rows.append(run_agent_benchmark(agent_name, agent, conversations, agent_config))
    print(f"\n## {title}\n")
    print(format_rows(rows))


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    with TemporaryDirectory(prefix="memory-agent-benchmark-") as temp_dir:
        state_root = Path(temp_dir)
        _run_suite("Standard Benchmark", standard, config, state_root)
        _run_suite("Long-Context Stress Benchmark", stress, config, state_root)


if __name__ == "__main__":
    main()
