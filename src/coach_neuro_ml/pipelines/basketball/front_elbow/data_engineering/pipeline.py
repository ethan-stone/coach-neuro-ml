from kedro.pipeline import node, Pipeline
from .nodes import gather_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=gather_data,
                 inputs=["params:front_elbow_class_mappings", "params:data_gather_params"],
                 outputs="gcs_basketball_front_elbow_primary",
                 name="gather_basketball_front_elbow_data")
        ]
    )
