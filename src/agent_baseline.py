from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: conversation memory exists only inside a single thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id  # Baseline deliberately has no user-level memory.
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _session_facts(self, session: SessionState) -> dict[str, str]:
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        return facts

    @staticmethod
    def _looks_like_recall(message: str) -> bool:
        lowered = message.casefold()
        return "?" in message or any(
            marker in lowered
            for marker in ("nhắc lại", "mình tên gì", "hiện tại mình", "bạn biết", "tóm tắt")
        )

    @staticmethod
    def _format_facts(facts: dict[str, str]) -> str:
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
        ordered = [key for key in labels if key in facts]
        return "\n".join(f"- {labels[key]}: {facts[key]}" for key in ordered)

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in session.messages)

        facts = self._session_facts(session)
        if self._looks_like_recall(message):
            response = self._format_facts(facts) if facts else "Mình chưa có thông tin đó trong thread hiện tại."
        else:
            response = "Mình đã ghi nhận thông tin này trong thread hiện tại."

        session.messages.append({"role": "assistant", "content": response})
        generated = estimate_tokens(response)
        session.token_usage += generated
        return {
            "response": response,
            "content": response,
            "generated_tokens": generated,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        response = result.content if hasattr(result, "content") else str(result)
        session.messages.append({"role": "assistant", "content": response})
        generated = estimate_tokens(response)
        session.token_usage += generated
        return {
            "response": response,
            "content": response,
            "generated_tokens": generated,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        provider = self.config.model.provider.strip().lower()
        has_connection = bool(self.config.model.api_key) or provider == "ollama"
        if not has_connection:
            return None
        try:
            return build_chat_model(self.config.model)
        except (RuntimeError, ValueError):
            return None
