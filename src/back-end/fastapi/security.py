import hmac
import logging
from io import BytesIO

from fastapi import Header, HTTPException, status
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("smart-vet")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    from main import get_settings

    configured_key = get_settings().api_key
    if configured_key and not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if configured_key and not hmac.compare_digest(x_api_key or "", configured_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


def validate_image_bytes(image_bytes: bytes) -> None:
    from main import get_settings

    settings = get_settings()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image cannot be empty")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the allowed size")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Rejected invalid image upload")
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
