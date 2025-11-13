from collections import deque


class ConversationMemory:
    def __init__(self, k: int = 6):
        self.buf = deque(maxlen=k)

    def note(self, user: str, assistant: str | None = None):
        self.buf.append((user, assistant))

    def hint(self) -> str:
        items = []
        for u, a in self.buf:
            if a:
                items.append(f"User: {u}\nAssistant: {a}")
            else:
                items.append(f"User: {u}")
        return "\n".join(items)
