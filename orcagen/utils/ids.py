from datetime import datetime


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_object_id(name: str) -> str:
    if name.startswith("Toys_"):
        name = name[len("Toys_") :]
    out = "".join([c for c in name.lower() if c.isalnum() or c == "_"])
    return out or name.lower()

