from .....extras.utilities import split_data_generic, train_model_generic, evaluate_model_generic


def split_data(data, parameters):
    return split_data_generic(data, parameters)


def train_model(X_train, y_train):
    return train_model_generic(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    return evaluate_model_generic(model, X_test, y_test, "BasketballFrontElbowModel")
