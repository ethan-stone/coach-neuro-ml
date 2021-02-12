from pathlib import PurePosixPath

from src.coach_neuro_ml.extras.datasets.gcs_json_dataset import GCSJSONDataSet


def test_proper_instantiation():
    dataset = GCSJSONDataSet("gcs://coachneuromlbucket/raw-json-data/basketball/front_elbow.json",
                             {
                                 "id_token": "C:/Users/Ethan/CoachNeuro/coach-neuro-ml/conf/local/ml-dev-storage-service-account.json"},
                             {"project": "coachneuro"})
    root = dataset._fs.ls("coachneuromlbucket")
    print(root)
    assert root == ['coachneuromlbucket/primary-csv-data', 'coachneuromlbucket/raw-json-data']


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

    dataset.save(df)
