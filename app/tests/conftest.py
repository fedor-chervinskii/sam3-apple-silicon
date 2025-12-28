"""Test configuration and fixtures for SAM3 API tests."""

import base64
import io
import os
from typing import Optional

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app as fastapi_app


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variable."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@pytest.fixture(scope="session")
def sam3_model():
    """Load SAM3 model once for all tests.
    
    Requires HF_TOKEN environment variable to access the gated SAM3 model.
    If not available, tests requiring the model will be skipped.
    """
    hf_token = get_hf_token()
    
    if not hf_token:
        pytest.skip(
            "SAM3 model loading skipped: HF_TOKEN not found in environment. "
            "Set HF_TOKEN environment variable with your HuggingFace token to run model tests. "
            "Get a token at https://huggingface.co/settings/tokens and request access to facebook/sam3"
        )
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("\nSuccessfully authenticated with HuggingFace")
    except Exception as e:
        pytest.skip(f"HuggingFace authentication failed: {e}")
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Get BPE path
    import sam3
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    # Load model
    print("Loading SAM3 model for tests...")
    try:
        from sam3 import build_sam3_image_model
        model = build_sam3_image_model(bpe_path=bpe_path)
        
        # Enable autocast for bfloat16
        if torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
        print("SAM3 model loaded successfully!")
        return model
    except Exception as e:
        pytest.skip(f"Failed to load SAM3 model: {e}")


@pytest.fixture(scope="session")
def app_with_model(sam3_model):
    """Get FastAPI app with model loaded."""
    from app.main import model_state
    model_state["model"] = sam3_model
    yield fastapi_app
    model_state.clear()


@pytest.fixture
def client(app_with_model):
    """Create a test client with model loaded."""
    return TestClient(app_with_model)


@pytest.fixture(scope="session")
def test_image_base64() -> str:
    """Create a simple test image and return as base64."""
    # Create a simple RGB image (100x100 red square)
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str


@pytest.fixture(scope="session")
def test_image_with_data_uri(test_image_base64: str) -> str:
    """Return test image with data URI prefix."""
    return f"data:image/png;base64,{test_image_base64}"


@pytest.fixture(scope="session")
def sample_text_request(test_image_base64: str) -> dict:
    """Sample request with text prompt."""
    return {
        "image": test_image_base64,
        "prompt": "object",
        "confidence_threshold": 0.5
    }


@pytest.fixture(scope="session")
def sample_box_request(test_image_base64: str) -> dict:
    """Sample request with box prompts."""
    return {
        "image": test_image_base64,
        "boxes": [
            {
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.3,
                "h": 0.3,
                "label": True
            }
        ],
        "confidence_threshold": 0.5
    }


@pytest.fixture(scope="session")
def sample_combined_request(test_image_base64: str) -> dict:
    """Sample request with both text and box prompts."""
    return {
        "image": test_image_base64,
        "prompt": "object",
        "boxes": [
            {
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.3,
                "h": 0.3,
                "label": True
            }
        ],
        "confidence_threshold": 0.3
    }
