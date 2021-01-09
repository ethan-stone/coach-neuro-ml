from kedro.pipeline import node, Pipeline
from .nodes import split_data, train_model, evaluate_model


def create_pipeline():
    return Pipeline(
        [
            node(func=split_data,
                 inputs=["aws_basketball_front_elbow_primary", "parameters"],
                 outputs=["X_train_basketball_front_elbow",
                          "X_test_basketball_front_elbow",
                          "y_train_basketball_front_elbow",
                          "y_test_basketball_front_elbow"],
                 name="split_basketball_front_elbow_data"),
            node(func=train_model,
                 inputs=["X_train_basketball_front_elbow",
                         "y_train_basketball_front_elbow"],
                 outputs="aws_basketball_front_elbow_model",
                 name="train_basketball_front_elbow_model"),
            node(func=evaluate_model,
                 inputs=["aws_basketball_front_elbow_model",
                         "X_test_basketball_front_elbow",
                         "y_test_basketball_front_elbow"],
                 outputs=None,
                 name="evaluate_basketball_front_elbow_model")
        ]
    )
