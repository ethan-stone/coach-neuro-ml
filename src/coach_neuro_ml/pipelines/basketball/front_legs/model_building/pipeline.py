from kedro.pipeline import node, Pipeline
from .nodes import split_data, train_model, evaluate_model


def create_pipeline():
    return Pipeline(
        [
            node(func=split_data,
                 inputs=["gcs_basketball_front_legs_primary", "params:split_data_params"],
                 outputs=["X_train_basketball_front_legs",
                          "X_test_basketball_front_legs",
                          "X_val_basketball_front_legs",
                          "y_val_basketball_front_legs",
                          "y_train_basketball_front_legs",
                          "y_test_basketball_front_legs"],
                 name="split_basketball_front_legs_data"),
            node(func=train_model,
                 inputs=["X_train_basketball_front_legs",
                         "y_train_basketball_front_legs",
                         "X_val_basketball_front_legs",
                         "y_val_basketball_front_legs",
                         "params:front_elbow_model_params"],
                 outputs="gcs_basketball_front_legs_model",
                 name="train_basketball_front_legs_model"),
            node(func=evaluate_model,
                 inputs=["gcs_basketball_front_legs_model",
                         "X_test_basketball_front_legs",
                         "y_test_basketball_front_legs"],
                 outputs=None,
                 name="evaluate_basketball_front_legs_model")
        ]
    )