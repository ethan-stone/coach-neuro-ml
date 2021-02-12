from pathlib import PurePosixPath
from src.coach_neuro_ml.extras.datasets.gcs_csv_dataset import GCSCSVDataSet


def test_proper_instantiation():
    dataset = GCSCSVDataSet("gcs://coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage"
                                            "-service-account.json "
                            },
                            {"project": "coachneuro"})
    root = dataset._fs.ls("coachneuromlbucket")
    assert root == ['coachneuromlbucket/primary-csv-data', 'coachneuromlbucket/raw-json-data']


def test_describe():
    dataset = GCSCSVDataSet("gcs://coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage"
                                            "-service-account.json "
                            },
                            {"project": "coachneuro"})
    description = dataset._describe()
    assert description == dict(filepath=PurePosixPath("coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv"))


def test_load():
    dataset = GCSCSVDataSet("gcs://coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage"
                                            "-service-account.json "
                            },
                            {"project": "coachneuro"})
    df = dataset.load()
    assert df is not None


def test_save():
    dataset = GCSCSVDataSet("gcs://coachneuromlbucket/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage"
                                            "-service-account.json "
                            },
                            {"project": "coachneuro"})
    df = dataset.load()

    dataset.save(df)
