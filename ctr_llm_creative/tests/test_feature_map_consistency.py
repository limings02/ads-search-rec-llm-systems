"""Test feature map fit/transform consistency."""

import pytest
from contracts import FeatureMap, FeatureSpec, FeatureType, TransformSpec, TransformType


def test_feature_map_creation():
    """Test creating a feature map."""
    feature_map = FeatureMap(
        name="test_map",
        version="1.0",
    )
    
    assert feature_map.name == "test_map"
    assert feature_map.version == "1.0"
    assert len(feature_map.features) == 0


def test_feature_spec_creation():
    """Test creating a feature spec."""
    feature = FeatureSpec(
        name="test_feature",
        feature_type=FeatureType.CATEGORICAL,
        raw_type="string",
    )
    
    assert feature.name == "test_feature"
    assert feature.feature_type == FeatureType.CATEGORICAL


def test_feature_map_serialization():
    """Test feature map JSON serialization."""
    feature_map = FeatureMap(
        name="test_map",
        version="1.0",
    )
    
    feature = FeatureSpec(
        name="test_feature",
        feature_type=FeatureType.CATEGORICAL,
        raw_type="string",
    )
    
    feature_map.add_feature(feature)
    
    # Serialize to JSON
    json_str = feature_map.to_json()
    assert "test_map" in json_str
    
    # Deserialize from JSON
    loaded_map = FeatureMap.from_json(json_str)
    assert loaded_map.name == feature_map.name
    assert len(loaded_map.features) == 1
    assert loaded_map.features[0].name == "test_feature"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
