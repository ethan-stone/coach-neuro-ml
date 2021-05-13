import gcsfs
from pathlib import PurePosixPath
from typing import Any, Dict
import tensorflow as tf
import os
from ..utilities import Net
from kedro.io.core import (
    AbstractVersionedDataSet,
    Version,
    get_protocol_and_path
)


class GCSTensorflowModelDataSet(AbstractVersionedDataSet):
    def __init__(self, filepath: str, version: Version = None, credentials: Dict[str, Any] = None,
                 fs_args: Dict[str, Any] = None, load_args: Dict[str, Any] = None):
        _, path = get_protocol_and_path(filepath)
        self._fs = gcsfs.GCSFileSystem(project=fs_args["project"], token=credentials["id_token"])
        self.load_args = load_args

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def _load(self) -> Net:
        load_path = self._get_load_path()
        with self._fs.open(str(load_path), model="rb") as f:
            model = tf.keras.models.load_model(str(load_path))
            return model

    def _save(self, model: tf.keras.Model):
        save_path = self._get_save_path()
        filename = os.path.basename(str(save_path))
        local_path = f"C:/Users/Ethan/CoachNeuro/coach-neuro-ml/temp/{filename}"
        model.save(local_path)
        with open(local_path, mode="rb") as local_file:
            with self._fs.open(str(save_path), mode="wb") as gcs_file:
                gcs_file.write(local_file.read())
        os.remove(local_path)
