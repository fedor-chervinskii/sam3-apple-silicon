"""FastAPI application for serving SAM3 model."""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Dict
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from app.models import SAM3Request, SAM3Response, ImageData
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# Global model instance
model_state: Dict = {}


def get_bpe_path() -> str:
    """Get the path to the BPE vocabulary file."""
    import sam3
    sam3_root = os.path.dirname(sam3.__file__)
    return f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"


def load_model():
    """Load SAM3 model and processor.
    
    Requires HF_TOKEN environment variable for accessing the gated SAM3 model.
    """
    import os
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable not found. "
            "SAM3 is a gated model that requires authentication. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and request access to facebook/sam3"
        )
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("Successfully authenticated with HuggingFace")
    except Exception as e:
        raise RuntimeError(f"HuggingFace authentication failed: {e}")
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    bpe_path = get_bpe_path()
    model = build_sam3_image_model(bpe_path=bpe_path)
    
    # Enable autocast for bfloat16
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading."""
    # Startup: load model
    print("Loading SAM3 model...")
    model_state["model"] = load_model()
    print("SAM3 model loaded successfully!")
    yield
    # Shutdown: cleanup
    model_state.clear()


# Create FastAPI app
app = FastAPI(
    title="SAM3 API",
    description="API for SAM 3 (Segment Anything Model 3) - text and visual prompting for image segmentation",
    version="0.1.0",
    lifespan=lifespan,
)


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG string."""
    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Encode as PNG
    success, buffer = cv2.imencode(".png", mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask as PNG")
    
    # Convert to base64
    base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return base64_str


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAM3 API",
        "version": "0.1.0",
        "endpoints": {
            "/sam3": "POST - Segment objects in images using text or visual prompts",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = "model" in model_state and model_state["model"] is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/sam3", response_model=SAM3Response)
async def segment_image(request: SAM3Request):
    """
    Segment objects in an image using SAM3.
    
    OpenAI-compatible API for image segmentation with text or visual prompts.
    Similar to OpenAI's image edit endpoint, but specialized for segmentation tasks.
    
    Supports:
    - Text prompts: Describe what to segment (e.g., "person", "face", "shoe")
    - Visual prompts: Provide bounding boxes as examples
    - Multiple results: Use 'n' parameter to get multiple mask variations
    """
    # Validate that at least one prompt is provided
    if not request.prompt and not request.boxes:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'prompt' (text) or 'boxes' (visual prompts) must be provided"
        )
    
    # Get model
    model = model_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode input image
        image = decode_base64_image(request.image)
        width, height = image.size
        
        # Create processor
        processor = Sam3Processor(model, confidence_threshold=request.confidence_threshold)
        inference_state = processor.set_image(image)
        
        # Apply prompts
        if request.prompt:
            # Text prompt
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=request.prompt
            )
        
        if request.boxes:
            # Visual prompts (bounding boxes)
            processor.reset_all_prompts(inference_state)
            for box in request.boxes:
                norm_box = [box.cx, box.cy, box.w, box.h]
                inference_state = processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=box.label
                )
        
        # Extract results
        data_list = []
        
        # Get masks from inference state
        if hasattr(inference_state, 'masks') and inference_state.masks is not None:
            masks = inference_state.masks
            scores = inference_state.scores if hasattr(inference_state, 'scores') else None
            
            # Handle tensor masks
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            
            if scores is not None and isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Process each mask
            num_masks = masks.shape[0] if len(masks.shape) > 2 else 1
            
            # Limit to requested number of results
            num_masks = min(num_masks, request.n) if request.n else num_masks
            
            for i in range(num_masks):
                if len(masks.shape) > 2:
                    mask = masks[i]
                else:
                    mask = masks
                
                # Get score
                score = float(scores[i]) if scores is not None and len(scores) > i else 0.5
                
                # Calculate bounding box from mask
                if len(mask.shape) == 3:
                    mask_2d = mask[0] if mask.shape[0] == 1 else mask.max(axis=0)
                else:
                    mask_2d = mask
                
                # Find bounding box
                rows = np.any(mask_2d, axis=1)
                cols = np.any(mask_2d, axis=0)
                
                if rows.any() and cols.any():
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                
                # Encode mask to base64
                mask_base64 = encode_mask_to_base64(mask_2d)
                
                # Create ImageData in OpenAI format
                data_list.append(
                    ImageData(
                        b64_json=mask_base64,
                        revised_prompt=request.prompt,
                        score=score,
                        bbox=bbox
                    )
                )
        
        # Create response in OpenAI format
        response = SAM3Response(
            created=int(time.time()),
            data=data_list
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
