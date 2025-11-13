from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile


class AICSImage:
    """
    Minimal reader that supports the small subset of the aicsimageio API used by
    ``pytorch_fnet``.  Data are assumed to be stored either as ``CZYX`` or
    ``ZYX``.  Additional scene or time dimensions are ignored because the test
    fixtures only exercise single-scene stacks.
    """

    def __init__(self, path: str | Path):
        self._path = str(path)
        with tifffile.TiffFile(self._path) as tif:
            series = tif.series[0]
            axes = (series.axes or self._infer_axes(series.shape)).upper()
            data = series.asarray()
        data, axes = self._drop_axis(data, list(axes), "S")
        data, axes = self._drop_axis(data, axes, "T")
        for axis in "CZYX":
            data, axes = self._ensure_axis(data, axes, axis)
        order = [axes.index(axis) for axis in "CZYX"]
        self._data = np.moveaxis(data, order, range(4))

    def __enter__(self) -> "AICSImage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get_image_data(
        self, out_orientation: str = "CZYX", *, S: int = 0, T: int = 0
    ) -> np.ndarray:
        if (S, T) != (0, 0):
            raise NotImplementedError("Scenes and timepoints are not supported.")

        orientation = out_orientation.upper()
        if orientation == "CZYX":
            return self._data
        if orientation == "ZYX":
            return self._data[0]
        raise NotImplementedError(f"Unsupported orientation: {out_orientation}")

    @property
    def data(self) -> np.ndarray:
        return self._data

    def size(self, dims: Sequence[str]) -> tuple[int, ...]:
        axis_map = {"C": 0, "Z": 1, "Y": 2, "X": 3}
        return tuple(self._data.shape[axis_map[d]] for d in dims)

    @staticmethod
    def _infer_axes(shape: Sequence[int]) -> str:
        lookup = {4: "CZYX", 3: "ZYX", 2: "YX", 1: "X"}
        return lookup.get(len(shape), "CZYX")

    @staticmethod
    def _drop_axis(
        data: np.ndarray, axes: list[str], axis: str, index: int = 0
    ) -> tuple[np.ndarray, list[str]]:
        if axis in axes:
            pos = axes.index(axis)
            data = np.take(data, index, axis=pos)
            axes.pop(pos)
        return data, axes

    @staticmethod
    def _ensure_axis(
        data: np.ndarray, axes: list[str], axis: str
    ) -> tuple[np.ndarray, list[str]]:
        if axis not in axes:
            data = np.expand_dims(data, axis=data.ndim)
            axes.append(axis)
        return data, axes
