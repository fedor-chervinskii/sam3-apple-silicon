# SAM3 API

FastAPI service for SAM3 (Segment Anything Model 3) with an OpenAI-compatible API format.

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-compatible GPU (recommended)
- **HuggingFace Account with SAM3 Access**

### HuggingFace Authentication

SAM3 is a gated model on HuggingFace. You need to:

1. Create a HuggingFace account at https://huggingface.co/join
2. Request access to the model: https://huggingface.co/facebook/sam3
3. Create an access token: https://huggingface.co/settings/tokens
4. Set the token as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### Installation

1. Install dependencies using uv:

```bash
cd app
uv pip install -e ..  # Install SAM3 package
uv pip install -r ../pyproject.toml  # Install API dependencies
```

Or install from the app directory:

```bash
cd app
uv pip install fastapi uvicorn[standard] pydantic pillow opencv-python numpy torch torchvision
```

### Running the API

Start the FastAPI server using uv:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with hot reload for development:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Usage

### Endpoint: POST /sam3

Segment objects in an image using text or visual prompts. Compatible with OpenAI's image API format.

#### Request Format

```bash
curl -X POST http://localhost:8000/sam3 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-image>",
    "prompt": "person",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded input image (PNG, JPEG, etc.) |
| `prompt` | string | No* | Text description of what to segment (e.g., "person", "face") |
| `boxes` | array | No* | Bounding box prompts in normalized coordinates |
| `n` | integer | No | Number of results to return (1-10, default: 1) |
| `size` | string | No | Output size (default: "1024x1024") |
| `response_format` | string | No | Always "b64_json" for base64-encoded masks |
| `model` | string | No | Model name (default: "sam3") |
| `confidence_threshold` | float | No | Confidence threshold 0.0-1.0 (default: 0.5) |
| `user` | string | No | Unique identifier for your end-user |

\* At least one of `prompt` or `boxes` must be provided

#### Response Format

```json
{
  "created": 1640000000,
  "data": [
    {
      "b64_json": "<base64-encoded-mask>",
      "revised_prompt": "person",
      "score": 0.95,
      "bbox": [100.0, 200.0, 300.0, 400.0]
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `created` | integer | Unix timestamp when masks were created |
| `data` | array | List of segmentation results |
| `data[].b64_json` | string | Base64-encoded mask image (PNG) |
| `data[].revised_prompt` | string | The prompt used for this result |
| `data[].score` | float | Confidence score (0.0-1.0) |
| `data[].bbox` | array | Bounding box [x, y, width, height] in pixels |

### Python Example

```python
import base64
import requests
from PIL import Image
import io

# Load and encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call API
response = requests.post(
    "http://localhost:8000/sam3",
    json={
        "image": image_b64,
        "prompt": "person",
        "n": 1
    }
)

# Decode mask
result = response.json()
if result["data"]:
    mask_b64 = result["data"][0]["b64_json"]
    mask_bytes = base64.b64decode(mask_b64)
    mask = Image.open(io.BytesIO(mask_bytes))
    mask.save("output_mask.png")
```

### Using Box Prompts

```json
{
  "image": "<base64-encoded-image>",
  "boxes": [
    {
      "cx": 0.5,
      "cy": 0.5,
      "w": 0.3,
      "h": 0.3,
      "label": true
    }
  ]
}
```

Box coordinates are normalized (0.0-1.0) in `[cx, cy, w, h]` format where:
- `cx`, `cy`: Center coordinates
- `w`, `h`: Width and height
- `label`: `true` for positive prompt, `false` for negative

## Testing

### Running Tests

The test suite includes tests that work without the model and tests that require the SAM3 model.

#### Tests without model (unit tests):
```bash
cd app
uv run pytest tests/test_models.py tests/test_utils.py -v
```

#### All tests (requires HF_TOKEN):
```bash
export HF_TOKEN=your_huggingface_token_here
cd app
uv run pytest tests/ -v
```

Run with coverage:

```bash
uv run pytest tests/ --cov=app --cov-report=html
```

### CI/CD Integration

For GitHub Actions, add your HuggingFace token as a repository secret:

1. Go to your repository Settings → Secrets and variables → Actions
2. Add a new secret named `HF_TOKEN`
3. Paste your HuggingFace token as the value

The included `.github/workflows/test.yml` will:
- Always run model-independent tests
- Run full tests with model if `HF_TOKEN` secret is configured

## Health Check

Check if the API is running and model is loaded:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true
}
```

## Notes

- First request may be slow as the model loads
- GPU is recommended for better performance
- Base64-encoded images should be < 50MB
- Supported image formats: PNG, JPEG, WebP
- The API follows OpenAI's image API conventions for compatibility
