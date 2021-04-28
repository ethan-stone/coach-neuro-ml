from kedro.pipeline import node, Pipeline
from .nodes import gather_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=gather_data,
                 inputs=["parameters"],
                 outputs=["gcs_basketball_front_elbow_primary"],
                 name="gather_basketball_front_elbow_data")
        ]
    )
