from kedro.pipeline import node, Pipeline
from .nodes import process_raw_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=process_raw_data,
                 inputs="aws_basketball_front_elbow_raw",
                 outputs="aws_basketball_front_elbow_primary",
                 name="process_basketball_front_elbow_raw")
        ]
    )
