"""Tests for request/response models."""

import pytest
from pydantic import ValidationError

from app.models import SAM3Request, SAM3Response, ImageData, PromptBox


class TestPromptBox:
    """Tests for PromptBox model."""
    
    def test_valid_prompt_box(self):
        """Test creating a valid PromptBox."""
        box = PromptBox(cx=0.5, cy=0.5, w=0.3, h=0.3, label=True)
        assert box.cx == 0.5
        assert box.cy == 0.5
        assert box.w == 0.3
        assert box.h == 0.3
        assert box.label is True
    
    def test_prompt_box_default_label(self):
        """Test that label defaults to True."""
        box = PromptBox(cx=0.5, cy=0.5, w=0.3, h=0.3)
        assert box.label is True
    
    def test_prompt_box_invalid_coordinates(self):
        """Test that invalid coordinates are rejected."""
        # cx > 1.0
        with pytest.raises(ValidationError):
            PromptBox(cx=1.5, cy=0.5, w=0.3, h=0.3)
        
        # cy < 0.0
        with pytest.raises(ValidationError):
            PromptBox(cx=0.5, cy=-0.1, w=0.3, h=0.3)
        
        # w > 1.0
        with pytest.raises(ValidationError):
            PromptBox(cx=0.5, cy=0.5, w=1.5, h=0.3)


class TestSAM3Request:
    """Tests for SAM3Request model."""
    
    def test_valid_text_request(self):
        """Test creating a valid request with text prompt."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person"
        )
        assert request.image == "base64encodedstring"
        assert request.prompt == "person"
        assert request.confidence_threshold == 0.5  # default
    
    def test_valid_box_request(self):
        """Test creating a valid request with box prompts."""
        request = SAM3Request(
            image="base64encodedstring",
            boxes=[
                PromptBox(cx=0.5, cy=0.5, w=0.3, h=0.3)
            ]
        )
        assert request.image == "base64encodedstring"
        assert len(request.boxes) == 1
    
    def test_combined_prompts(self):
        """Test request with both text and box prompts."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person",
            boxes=[PromptBox(cx=0.5, cy=0.5, w=0.3, h=0.3)]
        )
        assert request.prompt == "person"
        assert len(request.boxes) == 1
    
    def test_custom_confidence_threshold(self):
        """Test custom confidence threshold."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person",
            confidence_threshold=0.7
        )
        assert request.confidence_threshold == 0.7
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold is rejected."""
        with pytest.raises(ValidationError):
            SAM3Request(
                image="base64encodedstring",
                prompt="person",
                confidence_threshold=1.5
            )
    
    def test_missing_image(self):
        """Test that missing image is rejected."""
        with pytest.raises(ValidationError):
            SAM3Request(prompt="person")
    
    def test_n_parameter(self):
        """Test n parameter for number of results."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person",
            n=3
        )
        assert request.n == 3
    
    def test_response_format(self):
        """Test response_format parameter."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person",
            response_format="b64_json"
        )
        assert request.response_format == "b64_json"
    
    def test_model_parameter(self):
        """Test model parameter."""
        request = SAM3Request(
            image="base64encodedstring",
            prompt="person",
            model="sam3"
        )
        assert request.model == "sam3"


class TestImageData:
    """Tests for ImageData model."""
    
    def test_valid_image_data(self):
        """Test creating a valid ImageData."""
        result = ImageData(
            b64_json="base64mask",
            score=0.95,
            bbox=[100.0, 200.0, 300.0, 400.0]
        )
        assert result.b64_json == "base64mask"
        assert result.score == 0.95
        assert result.bbox == [100.0, 200.0, 300.0, 400.0]
    
    def test_minimal_image_data(self):
        """Test creating ImageData with only required field."""
        result = ImageData(b64_json="base64mask")
        assert result.b64_json == "base64mask"
        assert result.score is None
        assert result.bbox is None
    
    def test_with_revised_prompt(self):
        """Test ImageData with revised_prompt."""
        result = ImageData(
            b64_json="base64mask",
            revised_prompt="person in red shirt",
            score=0.9
        )
        assert result.revised_prompt == "person in red shirt"


class TestSAM3Response:
    """Tests for SAM3Response model."""
    
    def test_valid_response(self):
        """Test creating a valid SAM3Response."""
        response = SAM3Response(
            created=1640000000,
            data=[
                ImageData(
                    b64_json="base64mask",
                    score=0.95,
                    bbox=[100.0, 200.0, 300.0, 400.0]
                )
            ]
        )
        assert response.created == 1640000000
        assert len(response.data) == 1
    
    def test_empty_data(self):
        """Test response with no data."""
        response = SAM3Response(
            created=1640000000,
            data=[]
        )
        assert response.created == 1640000000
        assert len(response.data) == 0
    
    def test_multiple_results(self):
        """Test response with multiple results."""
        response = SAM3Response(
            created=1640000000,
            data=[
                ImageData(b64_json="mask1", score=0.95, bbox=[0, 0, 100, 100]),
                ImageData(b64_json="mask2", score=0.85, bbox=[200, 200, 150, 150])
            ]
        )
        assert len(response.data) == 2
