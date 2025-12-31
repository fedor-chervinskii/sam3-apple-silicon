"""Pydantic models for SAM3 API requests and responses."""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class PromptBox(BaseModel):
    """Bounding box prompt in normalized coordinates [cx, cy, w, h]."""
    cx: float = Field(..., ge=0.0, le=1.0, description="Normalized center x coordinate")
    cy: float = Field(..., ge=0.0, le=1.0, description="Normalized center y coordinate")
    w: float = Field(..., ge=0.0, le=1.0, description="Normalized width")
    h: float = Field(..., ge=0.0, le=1.0, description="Normalized height")
    label: bool = Field(default=True, description="Positive (True) or negative (False) box")


class SAM3Request(BaseModel):
    """Request model for SAM3 segmentation endpoint (OpenAI-compatible format)."""
    
    image: str = Field(
        ...,
        description="Base64-encoded input image (PNG, JPEG, etc.)"
    )
    prompt: Optional[str] = Field(
        None,
        description="Text prompt describing what to segment (e.g., 'person', 'face', 'shoe')"
    )
    mask: Optional[str] = Field(
        None,
        description="Base64-encoded mask image (PNG with transparency) indicating where to focus segmentation"
    )
    model: Optional[str] = Field(
        "sam3",
        description="Model to use for segmentation"
    )
    n: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Maximum number of masks to return (default: return all)"
    )
    size: Optional[str] = Field(
        "1024x1024",
        description="Output size in format WIDTHxHEIGHT (e.g., '1024x1024', '512x512')"
    )
    response_format: Optional[Literal["b64_json"]] = Field(
        "b64_json",
        description="Format of the response. Always 'b64_json' for base64-encoded masks"
    )
    boxes: Optional[List[PromptBox]] = Field(
        None,
        description="List of bounding box prompts in normalized coordinates [cx, cy, w, h] (SAM3-specific)"
    )
    confidence_threshold: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections (SAM3-specific)"
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing your end-user"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "prompt": "person",
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json"
            }
        }
    )


class ImageData(BaseModel):
    """Single image data result (OpenAI-compatible format)."""
    b64_json: str = Field(..., description="Base64-encoded segmentation mask (PNG)")
    revised_prompt: Optional[str] = Field(None, description="The prompt used for this result")
    
    # SAM3-specific fields
    score: Optional[float] = Field(None, description="Confidence score for this mask (SAM3-specific)")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x, y, w, h] in pixel coordinates (SAM3-specific)")


class SAM3Response(BaseModel):
    """Response model for SAM3 segmentation endpoint (OpenAI-compatible format)."""
    
    created: int = Field(..., description="Unix timestamp (in seconds) of when the masks were created")
    data: List[ImageData] = Field(..., description="List of generated segmentation masks")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "created": 1640000000,
                "data": [
                    {
                        "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        "score": 0.95,
                        "bbox": [100.0, 200.0, 300.0, 400.0]
                    }
                ]
            }
        }
    )
