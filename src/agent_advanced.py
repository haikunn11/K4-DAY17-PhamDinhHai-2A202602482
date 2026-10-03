from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term, persistent profile, and compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _prepare_turn(self, user_id: str, thread_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

    def _finish_turn(self, thread_id: str, response: str) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", response)
        generated = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        return {
            "response": response,
            "content": response,
            "generated_tokens": generated,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
        }

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._prepare_turn(user_id, thread_id, message)
        response = self._offline_response(user_id, thread_id, message)
        return self._finish_turn(thread_id, response)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        return (
            estimate_tokens(profile)
            + estimate_tokens(str(context["summary"]))
            + sum(estimate_tokens(str(item["content"])) for item in messages)
        )

    @staticmethod
    def _looks_like_recall(message: str) -> bool:
        lowered = message.casefold()
        return "?" in message or any(
            marker in lowered
            for marker in ("nhắc lại", "mình tên gì", "hiện tại mình", "bạn biết", "tóm tắt")
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        del thread_id
        facts = self.profile_store.facts(user_id)
        if not self._looks_like_recall(message):
            return "Mình đã cập nhật những thông tin ổn định và giữ ngữ cảnh gần đây."
        if not facts:
            return "Mình chưa có đủ thông tin bền vững về bạn."
        labels = {
            "name": "Tên",
            "profession": "Nghề nghiệp hiện tại",
            "location": "Nơi ở hiện tại",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "interests": "Mối quan tâm",
            "response_style": "Style trả lời",
        }
        return "\n".join(f"- {labels[key]}: {facts[key]}" for key in labels if key in facts)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._prepare_turn(user_id, thread_id, message)
        context = self.compact_memory.context(thread_id)
        prompt = [
            {"role": "system", "content": "Use the profile and compact memory below.\n" + self.profile_store.read_text(user_id) + "\nSummary: " + str(context["summary"])},
            *context["messages"],
        ]
        result = self.langchain_agent.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)
        return self._finish_turn(thread_id, response)

    def _maybe_build_langchain_agent(self):
        provider = self.config.model.provider.strip().lower()
        has_connection = bool(self.config.model.api_key) or provider == "ollama"
        if not has_connection:
            return None
        try:
            return build_chat_model(self.config.model)
        except (RuntimeError, ValueError):
            return None
