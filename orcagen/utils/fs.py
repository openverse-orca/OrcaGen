import os


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)

