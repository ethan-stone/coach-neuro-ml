import gcsfs
from pathlib import PurePosixPath
from typing import Any, Dict
from kedro.io.core import (
    AbstractDataSet,
    Version,
    get_protocol_and_path,
    get_filepath_str
)
import pandas as pd
import os


class GCSJSONDataSet(AbstractDataSet):
    def __init__(self, filepath: str, credentials: Dict[str, Any], fs_args: Dict[str, Any]):
        _, path = get_protocol_and_path(filepath)
        self._fs = gcsfs.GCSFileSystem(project=fs_args["project"], token=credentials["id_token"])
        self._filepath = PurePosixPath(path)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def _load(self) -> pd.DataFrame:
        print(str(self._filepath))
        with self._fs.open(str(self._filepath), mode="rb") as f:
            df = pd.read_json(f)
            return df

    def _save(self, data: pd.DataFrame) -> None:
        filename = os.path.basename(self._filepath)
        data.to_json(f"C:/Users/Ethan/CoachNeuro/coach-neuro-ml/temp{filename}")
        with open(f"C:/Users/Ethan/CoachNeuro/coach-neuro-ml/temp{filename}", "r") as local_file:
            with self._fs.open(str(self._filepath), mode="w") as gcs_file:
                gcs_file.write(local_file.read())
