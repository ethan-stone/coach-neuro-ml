from kedro.pipeline import node, Pipeline
from .nodes import split_data, train_model, evaluate_model


def create_pipeline():
    return Pipeline(
        [
            node(func=split_data,
                 inputs=["gcs_basketball_front_elbow_primary", "params:split_data_params"],
                 outputs=["X_train_basketball_front_elbow",
                          "X_test_basketball_front_elbow",
                          "X_val_basketball_front_elbow",
                          "y_val_basketball_front_elbow",
                          "y_train_basketball_front_elbow",
                          "y_test_basketball_front_elbow"],
                 name="split_basketball_front_elbow_data"),
            node(func=train_model,
                 inputs=["X_train_basketball_front_elbow",
                         "y_train_basketball_front_elbow",
                         "X_val_basketball_front_elbow",
                         "y_val_basketball_front_elbow",
                         "params:front_elbow_model_params"],
                 outputs="gcs_basketball_front_elbow_model",
                 name="train_basketball_front_elbow_model"),
            node(func=evaluate_model,
                 inputs=["gcs_basketball_front_elbow_model",
                         "X_test_basketball_front_elbow",
                         "y_test_basketball_front_elbow"],
                 outputs=None,
                 name="evaluate_basketball_front_elbow_model")
        ]
    )
