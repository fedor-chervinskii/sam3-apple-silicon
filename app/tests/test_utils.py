"""Unit tests for utility functions."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.main import decode_base64_image, encode_mask_to_base64


class TestImageDecoding:
    """Tests for base64 image decoding."""
    
    def test_decode_valid_base64(self):
        """Test decoding a valid base64 image."""
        # Create a simple image
        img = Image.new('RGB', (50, 50), color='blue')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Decode
        decoded = decode_base64_image(img_str)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (50, 50)
        assert decoded.mode == "RGB"
    
    def test_decode_with_data_uri(self):
        """Test decoding base64 with data URI prefix."""
        # Create a simple image
        img = Image.new('RGB', (50, 50), color='green')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Add data URI prefix
        data_uri = f"data:image/png;base64,{img_str}"
        
        # Decode
        decoded = decode_base64_image(data_uri)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (50, 50)
    
    def test_decode_invalid_base64(self):
        """Test that invalid base64 raises HTTPException."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            decode_base64_image("invalid_base64!!!")
        
        assert exc_info.value.status_code == 400
        assert "base64" in str(exc_info.value.detail).lower()
    
    def test_decode_converts_to_rgb(self):
        """Test that decoded images are converted to RGB."""
        # Create a grayscale image
        img = Image.new('L', (50, 50), color=128)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Decode
        decoded = decode_base64_image(img_str)
        assert decoded.mode == "RGB"


class TestMaskEncoding:
    """Tests for mask encoding to base64."""
    
    def test_encode_binary_mask(self):
        """Test encoding a binary mask to base64."""
        # Create a simple binary mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        
        # Encode
        encoded = encode_mask_to_base64(mask)
        
        # Verify it's valid base64
        assert isinstance(encoded, str)
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) > 0
    
    def test_encode_uint8_mask(self):
        """Test encoding a uint8 mask to base64."""
        # Create a uint8 mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1
        
        # Encode
        encoded = encode_mask_to_base64(mask)
        
        # Verify it's valid base64
        assert isinstance(encoded, str)
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) > 0
    
    def test_encoded_mask_is_png(self):
        """Test that encoded mask is a valid PNG."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        
        encoded = encode_mask_to_base64(mask)
        
        # Decode and verify it's a valid PNG
        decoded_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded_bytes))
        assert img.format == "PNG"
        assert img.size == (50, 50)
