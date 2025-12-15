import pytest
from pydantic import ValidationError
from spectrum.infra.types import RiskProfile, InferenceEvent, RiskLevel
from spectrum.infra.privacy import PIISanitizer

import spectrum.blue
# --------------------------
# 1. RiskProfile
# --------------------------

def test_risk_profile_valid():
    """Tests that a valid profile is created correctly."""
    rp = RiskProfile(level = RiskLevel.HIGH, alpha = 0.05)
    assert rp.alpha == 0.05
    assert rp.level == RiskLevel.HIGH

def test_risk_profile_invalid_alpha():
    """Tests that alpha outside (0, 1) raises a ValidationError."""
    with pytest.raises(ValidationError):
        RiskProfile(level=RiskLevel.LOW, alpha=1.5)
    with pytest.raises(ValidationError):
        RiskProfile(level=RiskLevel.LOW, alpha=-0.1)

# --------------------------
# InferenceEvent
# --------------------------

def test_inference_event_metadata_text():
    """Test that string inputs generate correct metadata."""
    payload = "This is a prompt injection attack."
    event = InferenceEvent(
        model_version="v1.0",
        input_payload=payload
    )
    
    assert event.data_type == "string"
    assert event.input_shape == [len(payload)]
    assert event.input_bytes > 0

# --------------------------
# PIISanitizer
# --------------------------

def test_sanitize_pii():
    """Test that PII are stripped from nested dicts."""
    sanitizer = PIISanitizer()
    payload = {
        "user_id": 123,
        "meta": {
            "contact": "contact@google.com",
            "ssn": "123-45-6789",
            "phone": "(773)-123-4567",
            "safe": "just_a_string"
        }
    }
    clean_data = sanitizer.sanitize(payload)
    
    assert clean_data["meta"]["contact"] == "[EMAIL]"
    assert clean_data["meta"]["ssn"] == "[SSN]"
    assert clean_data["meta"]["phone"] == "[PHONE]"
    assert clean_data["user_id"] == 123
    
def test_sanitize_tensor_boundary():
    """
    Test that 'Tensor-like' objects are summarized, not copied.
    We use a Fake class to avoid needing numpy/torch installed for tests.
    """
    class FakeTensor:
        def __init__(self):
            self.shape = (32, 1024)
            self.dtype = "float32"
            
    sanitizer = PIISanitizer()
    fake_tensor = FakeTensor()
    
    result = sanitizer.sanitize(fake_tensor)
    
    assert isinstance(result, str)
    assert "FakeTensor" in result
    assert "(32, 1024)" in result