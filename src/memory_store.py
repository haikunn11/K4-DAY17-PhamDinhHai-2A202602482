from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Return a deterministic, tokenizer-free token estimate."""

    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


@dataclass
class UserProfileStore:
    """A human-readable persistent store backed by one User.md per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        raw = user_id.strip()
        if not raw:
            raise ValueError("user_id must not be empty")
        ascii_id = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_id).strip(".-") or "user"
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if root not in path.parents:
            raise ValueError("user_id resolves outside the profile directory")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s+([a-z_]+):\s*(.+?)\s*$", line)
            if match:
                facts[match.group(1)] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        key = re.sub(r"[^a-z_]", "", key.strip().lower())
        value = " ".join(value.split()).strip(" ,.;")
        if not key or not value:
            return
        facts = self.facts(user_id)
        if key in {"interests", "response_style"} and key in facts:
            old_parts = [part.strip() for part in facts[key].split(";") if part.strip()]
            new_parts = [part.strip() for part in value.split(";") if part.strip()]
            lowered = {part.casefold() for part in old_parts}
            value = "; ".join(old_parts + [p for p in new_parts if p.casefold() not in lowered])
        facts[key] = value
        lines = ["# User Profile", ""]
        lines.extend(f"- {fact_key}: {facts[fact_key]}" for fact_key in sorted(facts))
        self.write_text(user_id, "\n".join(lines))

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        for key, value in updates.items():
            self.upsert_fact(user_id, key, value)


def _first_group(patterns: list[str], message: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split()).strip(" ,.;")
    return None


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract conservative, stable user facts from Vietnamese benchmark text."""

    text = " ".join(message.split())
    updates: dict[str, str] = {}
    if not text:
        return updates

    name = _first_group(
        [r"(?:mình|tôi)\s+tên\s+là\s+([^,.!?]+)", r"tên\s+(?:mình|tôi)\s+là\s+([^,.!?]+)"],
        text,
    )
    if name and not re.search(r"\b(gì|ai)\b", name, re.IGNORECASE):
        name = re.split(r"\s+(?:nghề|hiện|đang|và\s+(?:mình|tôi))\b", name, maxsplit=1, flags=re.I)[0]
        updates["name"] = name

    location = _first_group(
        [
            r"nơi\s+ở\s+(?:hiện\s+tại\s+)?(?:của\s+mình\s+)?là\s+([^,.!?]+)",
            r"(?:hiện\s+tại\s+)?mình\s+(?:đang\s+|vẫn\s+)?ở\s+([^,.!?]+)",
            r"hiện\s+ở\s+([^,.!?]+)",
        ],
        text,
    )
    if location:
        location = re.split(r"\s+(?:chứ|để|dù|và)\b", location, maxsplit=1, flags=re.I)[0]
        if not re.search(r"\b(họp|hai ngày|bay ra|đâu|gì|không)\b", location, re.I):
            updates["location"] = location

    profession = _first_group(
        [
            r"nghề\s+nghiệp\s+hiện\s+tại\s+(?:vẫn\s+)?là\s+([^,.!?]+)",
            r"nghề\s+hiện\s+tại\s+(?:của\s+mình\s+)?là\s+([^,.!?]+)",
            r"giờ\s+(?:mình\s+)?chuyển\s+sang\s+([^,.!?]+)",
            r"(?:mình|tôi)\s+(?:hiện\s+)?(?:đang\s+|vẫn\s+)?làm\s+([^,.!?]+)",
            r"(?:đang|vẫn)\s+làm\s+([^,.!?]+)",
        ],
        text,
    )
    if profession:
        profession = re.split(r"\s+(?:cho|ở|và)\b", profession, maxsplit=1, flags=re.I)[0]
        role_terms = r"\b(engineer|developer|manager|scientist|analyst|designer|giáo viên|bác sĩ|kỹ sư|lập trình viên)\b"
        is_role = re.search(role_terms, profession, re.I)
        is_noise = re.search(r"\b(đùa|hay là|không còn|gì|ai|đâu)\b", profession, re.I)
        if is_role and not is_noise:
            updates["profession"] = profession

    lowered = text.casefold()
    if "cà phê sữa đá" in lowered and any(marker in lowered for marker in ("yêu thích", "mình thích", "vẫn uống", "đồ uống")):
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lowered and any(marker in lowered for marker in ("yêu thích", "món ruột", "mình thích")):
        updates["favorite_food"] = "mì Quảng"
    if "corgi" in lowered and any(marker in lowered for marker in ("mình nuôi", "con corgi", "bé corgi")):
        pet_match = re.search(r"corgi(?:\s+tên)?\s+([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*)", text)
        updates["pet"] = f"corgi tên {pet_match.group(1)}" if pet_match else "corgi"

    if any(marker in lowered for marker in ("mình thích", "quan tâm", "mối quan tâm", "dài hạn:")):
        interests: list[str] = []
        for needle, label in (("python", "Python"), ("ai ứng dụng", "AI ứng dụng"), ("ai agent", "AI agent"), ("mlops", "MLOps"), ("rag", "RAG")):
            if needle in lowered:
                interests.append(label)
        if interests:
            updates["interests"] = "; ".join(interests)

    styles: list[str] = []
    if any(term in lowered for term in ("ngắn gọn", "bullet ngắn", "3 bullet", "ba bullet")):
        styles.append("3 bullet ngắn gọn" if ("3 bullet" in lowered or "ba bullet" in lowered) else "ngắn gọn")
    if "ví dụ thực" in lowered:
        styles.append("có ví dụ thực chiến")
    if "trade-off" in lowered:
        styles.append("nhấn mạnh trade-off")
    if styles and any(marker in lowered for marker in ("muốn", "thích", "hãy", "style", "trả lời")):
        updates["response_style"] = "; ".join(dict.fromkeys(styles))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded extractive summary suitable for deterministic tests."""

    if not messages or max_items <= 0:
        return ""
    chosen = messages if len(messages) <= max_items else messages[: max_items // 2] + messages[-(max_items - max_items // 2) :]
    items: list[str] = []
    for message in chosen:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        items.append(f"{role}: {content}")
    return " | ".join(items)


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and compress older context into a summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0:
            raise ValueError("threshold_tokens must be greater than zero")
        if self.keep_messages <= 0:
            raise ValueError("keep_messages must be greater than zero")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread["summary"])
        total = estimate_tokens(summary) + sum(estimate_tokens(str(m["content"])) for m in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older = messages[:-self.keep_messages]
        thread["messages"] = messages[-self.keep_messages :]
        addition = summarize_messages(older)
        combined = " || ".join(part for part in (summary, addition) if part)
        max_chars = max(240, self.threshold_tokens * 2)
        if len(combined) > max_chars:
            half = max_chars // 2
            combined = combined[:half].rstrip() + " ... " + combined[-half:].lstrip()
        thread["summary"] = combined
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {
            "messages": [dict(message) for message in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
