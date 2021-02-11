from pathlib import PurePosixPath

from src.coach_neuro_ml.extras.datasets.gcs_json_dataset import GCSJSONDataSet
from src.coach_neuro_ml.pipelines.utilities import process_raw_data_generic


def test_proper_instantiation():
    dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.json",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    root = dataset._fs.ls("coachneuromlbucket")
    assert root == ["coachneuromlbucket/raw-json-data"]


def test_describe():
    dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.json",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    description = dataset._describe()
    assert description == dict(filepath=PurePosixPath("coachneuromlbucket/raw-json-data/basketball/front_elbow.json"))


def test_load():
    dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.json",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    df = dataset.load()
    assert df is not None


def test_save():
    dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.json",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    df = dataset.load()
    new_df = process_raw_data_generic(df, {"i": 0, "m": 1, "o": 2})

    dataset.save(new_df)

    new_dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.csv",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    loaded_new_df = new_dataset.load()
    assert loaded_new_df is not None
