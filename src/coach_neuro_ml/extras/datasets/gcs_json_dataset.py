import gcsfs
from pathlib import PurePosixPath
from typing import Any, Dict
from kedro.io.core import (
    AbstractVersionedDataSet,
    Version,
    get_protocol_and_path,
    get_filepath_str
)
import pandas as pd


class GCSJSONDataSet(AbstractVersionedDataSet):
    def __init__(self, filepath: str, credentials: Dict[str, Any], fs_args: Dict[str, Any], version: Version = None):
        _, path = get_protocol_and_path(filepath)
        self._fs = gcsfs.GCSFileSystem(project=fs_args["project"], token=credentials["id_token"])

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def _load(self) -> pd.DataFrame:
        load_path = self._get_load_path()
        with self._fs.open(str(load_path), mode="r") as f:
            df = pd.read_json(f)
            return df

    def _save(self, data: Any) -> None:
            return None
