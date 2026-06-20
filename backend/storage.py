import uuid
import tempfile
from pathlib import Path

_store: dict[str, dict] = {}


def save_upload(filename: str, data: bytes) -> str:
    media_id = str(uuid.uuid4())
    ext = filename.rsplit(".", 1)[-1].lower()
    path = Path(tempfile.gettempdir()) / f"offside_{media_id}.{ext}"
    path.write_bytes(data)
    _store[media_id] = {"path": str(path), "ext": ext}
    return media_id


def get_info(media_id: str) -> dict | None:
    return _store.get(media_id)
