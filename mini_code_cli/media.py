"""Image attachments for chat messages.

The OpenAI-compatible endpoint accepts image input as a content list:

    [{"type": "text", "text": "what is this?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

Base64 in the persisted history would make every session file huge and resend
the bytes on every request. Instead, a message holds a small ``image_ref``
block naming a file in the session cache::

    {"type": "image_ref", "path": "/cache/<session>/attachments/...",
     "mime": "image/png", "name": "board.png"}

:func:`materialize_messages` resolves those refs into data URIs only when a
request is built. Text-only consumers (plan/review/compaction prompts, token
estimates, history views) use :func:`content_text`, which turns an image ref
into a short ``[image: name]`` marker.
"""
import base64
import hashlib
import mimetypes
import os
import shutil

ATTACHMENT_DIRNAME = "attachments"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
#: Rough prompt cost of one image for the context estimate. The server's real
#: count depends on dimensions; this is deliberately conservative and only
#: affects the local projection used for compaction.
IMAGE_TOKEN_ESTIMATE = 1000

IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_mime(path):
    """Return the MIME type for a supported image path, or ``None``."""
    return IMAGE_MIMES.get(os.path.splitext(str(path or ""))[1].lower())


def is_image_path(path):
    return image_mime(path) is not None


def stage_image(path, session_dir, name=None):
    """Copy an image into the session cache and return an ``image_ref`` block.

    Raises ``ValueError`` for unsupported extensions, unreadable files and
    oversized images. The copy is content-addressed so attaching the same image
    twice reuses one file.
    """
    source = os.path.abspath(os.path.expanduser(str(path or "")))
    mime = image_mime(source)
    if not mime:
        raise ValueError(
            "unsupported image type; use PNG, JPEG, WEBP or GIF")
    try:
        size = os.path.getsize(source)
    except OSError as exc:
        raise ValueError(f"cannot read image: {exc}")
    if size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image is too large ({size} bytes; limit {MAX_IMAGE_BYTES})")
    try:
        with open(source, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read image: {exc}")

    digest = hashlib.sha1(data).hexdigest()[:16]
    base = _safe_name(name or os.path.basename(source))
    directory = os.path.join(session_dir, ATTACHMENT_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, f"{digest}-{base}")
    if not os.path.isfile(target):
        tmp = target + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    return {"type": "image_ref", "path": target, "mime": mime,
            "name": os.path.basename(source), "bytes": len(data)}


def _safe_name(name):
    name = os.path.basename(str(name or "image"))
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                     for ch in name)
    return cleaned or "image"


def user_content(text, image_refs=()):
    """Build a user message content value from text and image refs."""
    text = str(text or "")
    refs = [ref for ref in image_refs if isinstance(ref, dict)]
    if not refs:
        return text
    return [{"type": "text", "text": text}] + refs


def content_text(content):
    """Flatten a message content value to text for text-only consumers."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    parts.append(str(part.get("text") or ""))
                elif kind in ("image_ref", "image_url"):
                    name = part.get("name")
                    if not name:
                        name = os.path.basename(str(part.get("path") or "image"))
                    parts.append(f"[image: {name}]")
                elif kind:
                    parts.append(f"[{kind}]")
        return "\n".join(part for part in parts if part)
    return str(content)


def image_refs(content):
    """Return the image-ref blocks in a content value."""
    if not isinstance(content, list):
        return []
    return [part for part in content
            if isinstance(part, dict) and part.get("type") == "image_ref"]


def materialize_messages(messages):
    """Return messages with ``image_ref`` blocks resolved to data URIs.

    The original list is not mutated: history keeps the small file references.
    A missing attachment becomes a text marker rather than failing the request.
    """
    result = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            result.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            result.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_ref":
                changed = True
                resolved = _materialize_ref(part)
                parts.append(resolved)
            else:
                parts.append(part)
        new_message = dict(message)
        new_message["content"] = parts
        result.append(new_message)
    return result if changed else messages


def _materialize_ref(ref):
    path = ref.get("path")
    mime = ref.get("mime") or mimetypes.guess_type(str(path))[0]
    try:
        with open(str(path), "rb") as handle:
            data = handle.read()
    except OSError:
        label = ref.get("name") or os.path.basename(str(path or "image"))
        return {"type": "text",
                "text": f"[image unavailable: {label}]"}
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def copy_into_session(path, session_dir, name=None):
    """Compatibility wrapper used by the session command."""
    return stage_image(path, session_dir, name=name)


def remove_session_attachments(session_dir):
    """Remove the attachment directory (used by explicit session cleanup)."""
    shutil.rmtree(os.path.join(session_dir, ATTACHMENT_DIRNAME),
                  ignore_errors=True)
