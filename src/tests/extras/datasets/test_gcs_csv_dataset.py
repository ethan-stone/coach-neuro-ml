from pathlib import PurePosixPath
from src.coach_neuro_ml.extras.datasets.gcs_csv_dataset import GCSCSVDataSet


def test_describe():
    dataset = GCSCSVDataSet("gcs://coachneuro-dev-ml/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/coachneuro-dev-ml.json"
                            },
                            {"project": "coachneuro-dev"})
    description = dataset._describe()
    assert description == dict(filepath=PurePosixPath("coachneuro-dev-ml/primary-csv-data/basketball/front_elbow.csv"))


def test_load():
    dataset = GCSCSVDataSet("gcs://coachneuro-dev-ml/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/coachneuro-dev-ml.json"
                            },
                            {"project": "coachneuro-dev"})
    df = dataset.load()
    assert df is not None


def test_save():
    dataset = GCSCSVDataSet("gcs://coachneuro-dev-ml/primary-csv-data/basketball/front_elbow.csv",
                            {
                                "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/coachneuro-dev-ml.json"
                            },
                            {"project": "coachneuro-dev"})
    df = dataset.load()

    dataset.save(df)
