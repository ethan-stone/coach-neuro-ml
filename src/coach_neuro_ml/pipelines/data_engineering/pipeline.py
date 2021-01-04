from kedro.pipeline import node, Pipeline
from .nodes import process_raw_elbow_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=process_raw_elbow_data,
                 inputs="elbow_raw_dataset",
                 outputs="elbow_primary_dataset",
                 name="process_raw_elbow_data"
                 ),
        ]
    )
