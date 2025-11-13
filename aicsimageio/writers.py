from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


class OmeTiffWriter:
    """Very small subset of the upstream writer used in the tests."""

    def __init__(self, path: str | Path):
        self._path = str(path)

    def __enter__(self) -> "OmeTiffWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def save(
        self, data: np.ndarray, *, dimension_order: str = "CZYX", **kwargs
    ) -> None:
        arr = np.asarray(data)
        metadata = {"axes": dimension_order}
        tifffile.imwrite(
            self._path,
            arr,
            metadata=metadata,
            photometric="minisblack",
        )


__all__: Iterable[str] = ["OmeTiffWriter"]
