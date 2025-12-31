"""End-to-end test with toy data."""

import pytest
from contracts import (
    DatasetManifest,
    TaskSpec,
    TaskType,
    LabelSpec,
    SplitSpec,
    SplitType,
    FeatureMap,
    FeatureSpec,
    FeatureType,
    TransformSpec,
    TransformType,
)


def test_end2end_workflow():
    """Test a complete workflow: manifest -> features -> training."""
    
    # Step 1: Create dataset manifest
    manifest = DatasetManifest(
        name="toy_dataset",
        version="1.0",
        dataset_type="ctr",
        description="Toy dataset for testing",
        feature_fields=["feature_1", "feature_2", "feature_3"],
        label_fields=["click"],
        id_field="id",
        timestamp_field="time",
    )
    
    # Add task
    manifest.tasks.append(
        TaskSpec(
            name="ctr_task",
            task_type=TaskType.CTR,
            labels=[
                LabelSpec(
                    name="click",
                    task_type=TaskType.CTR,
                    positive_value=1,
                    negative_value=0,
                )
            ],
        )
    )
    
    # Add splits
    manifest.splits.extend([
        SplitSpec(
            name=SplitType.TRAIN,
            path="data/processed/train.parquet",
            size=1000,
        ),
        SplitSpec(
            name=SplitType.VALID,
            path="data/processed/valid.parquet",
            size=100,
        ),
        SplitSpec(
            name=SplitType.TEST,
            path="data/processed/test.parquet",
            size=100,
        ),
    ])
    
    assert len(manifest.tasks) == 1
    assert len(manifest.splits) == 3
    
    # Step 2: Create feature map
    feature_map = FeatureMap(
        name="toy_features",
        version="1.0",
        fitted_on="toy_dataset",
    )
    
    # Add features
    for i in range(1, 4):
        feature = FeatureSpec(
            name=f"feature_{i}",
            feature_type=FeatureType.CATEGORICAL,
            raw_type="string",
            transforms=[
                TransformSpec(
                    type=TransformType.HASH,
                    params={"size": 1000},
                )
            ],
        )
        feature_map.add_feature(feature)
    
    assert len(feature_map.features) == 3
    
    # Step 3: Serialize and deserialize
    manifest_json = manifest.to_json()
    feature_map_json = feature_map.to_json()
    
    loaded_manifest = DatasetManifest.from_json(manifest_json)
    loaded_feature_map = FeatureMap.from_json(feature_map_json)
    
    assert loaded_manifest.name == manifest.name
    assert loaded_feature_map.name == feature_map.name
    assert len(loaded_feature_map.features) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
