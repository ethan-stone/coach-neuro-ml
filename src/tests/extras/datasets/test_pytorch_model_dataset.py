from pathlib import PurePosixPath
from src.coach_neuro_ml.extras.datasets.gcs_tensorflow_model_dataset import GCSPyTorchModelDataSet
from src.coach_neuro_ml.extras.datasets.gcs_csv_dataset import GCSCSVDataSet
from src.coach_neuro_ml.pipelines.utilities import split_data_generic, train_model_generic, evaluate_model_generic
from kedro.io.core import Version


def test_describe():
    dataset = GCSPyTorchModelDataSet("gcs://coachneuromlbucket/models/basketball/front_legs.pt",
                                     version=Version(None, None),
                                     credentials={"id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local"
                                                              "/ml-dev-storage-service-account.json"},
                                     fs_args={"project": "coachneuro"})
    description = dataset._describe()
    assert description == dict(filepath=PurePosixPath("coachneuromlbucket/models/basketball/front_legs.pt"),
                               version=Version(None, None))


def test_save():
    pytorch_dataset = GCSPyTorchModelDataSet("gcs://coachneuromlbucket/models/basketball/front_legs.pt",
                                             version=Version(None, None),
                                             credentials={
                                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local"
                                                             "/ml-dev-storage-service-account.json"},
                                             fs_args={"project": "coachneuro"})

    dataset = GCSCSVDataSet("gcs://coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage"
                                            "-service-account.json "
                            },
                            {"project": "coachneuro"})

    df = dataset.load()

    X_train, X_test, y_train, y_test = split_data_generic(df, {"test_size": 0.2, "random_state": 69})
    model = train_model_generic(X_train, y_train)

    evaluate_model_generic(model, X_test, y_test, "BasketballFrontLegsModel")

    pytorch_dataset.save(model)


def test_load():

    dataset = GCSPyTorchModelDataSet("gcs://coachneuromlbucket/models/basketball/front_legs.pt",
                                     version=Version(None, None),
                                     credentials={"id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local"
                                                              "/ml-dev-storage-service-account.json"},
                                     fs_args={"project": "coachneuro"})
    model = dataset.load()
