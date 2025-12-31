"""Tests for SAM3 API endpoints."""

import base64

import pytest
from fastapi.testclient import TestClient


class TestRootEndpoints:
    """Tests for root and health endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "SAM3 API" in data["message"]
        assert "endpoints" in data
    
    def test_health_endpoint(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "cuda_available" in data


class TestSAM3Endpoint:
    """Tests for the /sam3 segmentation endpoint."""
    
    def test_sam3_endpoint_with_text_prompt(self, client: TestClient, sample_text_request: dict):
        """Test SAM3 endpoint with text prompt."""
        response = client.post("/sam3", json=sample_text_request)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "created" in data
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_sam3_endpoint_with_box_prompt(self, client: TestClient, sample_box_request: dict):
        """Test SAM3 endpoint with box prompts."""
        response = client.post("/sam3", json=sample_box_request)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "created" in data
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_sam3_endpoint_missing_prompts(self, client: TestClient, test_image_base64: str):
        """Test SAM3 endpoint fails when no prompts provided."""
        request = {
            "image": test_image_base64,
            "confidence_threshold": 0.5
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 400
        assert "prompt" in response.json()["detail"].lower() or "boxes" in response.json()["detail"].lower()
    
    def test_sam3_endpoint_invalid_base64(self, client: TestClient):
        """Test SAM3 endpoint with invalid base64 image."""
        request = {
            "image": "invalid_base64_string!!!",
            "prompt": "object"
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 400
        assert "base64" in response.json()["detail"].lower()
    
    def test_sam3_endpoint_with_data_uri(self, client: TestClient, test_image_with_data_uri: str):
        """Test SAM3 endpoint accepts data URI format."""
        request = {
            "image": test_image_with_data_uri,
            "prompt": "object"
        }
        response = client.post("/sam3", json=request)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    def test_sam3_endpoint_custom_confidence(self, client: TestClient, test_image_base64: str):
        """Test SAM3 endpoint with custom confidence threshold."""
        request = {
            "image": test_image_base64,
            "prompt": "object",
            "confidence_threshold": 0.3
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    def test_sam3_endpoint_invalid_confidence(self, client: TestClient, test_image_base64: str):
        """Test SAM3 endpoint with invalid confidence threshold."""
        request = {
            "image": test_image_base64,
            "prompt": "object",
            "confidence_threshold": 1.5  # Invalid: > 1.0
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_sam3_endpoint_combined_prompts(self, client: TestClient, sample_combined_request: dict):
        """Test SAM3 endpoint with both text and box prompts."""
        response = client.post("/sam3", json=sample_combined_request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    def test_sam3_endpoint_multiple_boxes(self, client: TestClient, test_image_base64: str):
        """Test SAM3 endpoint with multiple box prompts."""
        request = {
            "image": test_image_base64,
            "boxes": [
                {"cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2, "label": True},
                {"cx": 0.7, "cy": 0.7, "w": 0.2, "h": 0.2, "label": False}
            ]
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    def test_sam3_response_structure(self, client: TestClient, sample_text_request: dict):
        """Test that successful response has correct structure."""
        response = client.post("/sam3", json=sample_text_request)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Check top-level fields
        assert "created" in data
        assert "data" in data
        assert isinstance(data["created"], int)
        assert isinstance(data["data"], list)
        
        # Check data structure if any results returned
        if len(data["data"]) > 0:
            result = data["data"][0]
            assert "b64_json" in result
            assert isinstance(result["b64_json"], str)
            
            # SAM3-specific fields (optional)
            if "score" in result:
                assert isinstance(result["score"], float)
            if "bbox" in result:
                assert isinstance(result["bbox"], list)
                assert len(result["bbox"]) == 4
            
            # Verify b64_json is valid base64
            try:
                mask_bytes = base64.b64decode(result["b64_json"])
                assert len(mask_bytes) > 0
            except Exception:
                pytest.fail("b64_json is not valid base64")


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_box_coordinates_validation(self, client: TestClient, test_image_base64: str):
        """Test box coordinate validation."""
        # Test invalid cx (> 1.0)
        request = {
            "image": test_image_base64,
            "boxes": [{"cx": 1.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
        }
        response = client.post("/sam3", json=request)
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client: TestClient):
        """Test that missing required fields are caught."""
        response = client.post("/sam3", json={})
        assert response.status_code == 422
    
    def test_invalid_json(self, client: TestClient):
        """Test handling of invalid JSON."""
        response = client.post(
            "/sam3",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
