"""Generates predictions from a model."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader

from fnet.cli.init import save_default_predict_options
from fnet.data import FnetDataset, TiffDataset
from fnet.models import load_model
from fnet.transforms import norm_around_center
from fnet.utils.general_utils import files_from_dir
from fnet.utils.general_utils import retry_if_oserror
from fnet.utils.general_utils import str_to_object

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_mask_image, to_numpy

torch.backends.cudnn.enabled = True

logger = logging.getLogger(__name__)


def get_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    """Returns dataset.

    Returns
    -------
    torch.utils.data.Dataset
        Dataset object.

    """
    if sum([args.dataset is not None, args.path_tif is not None]) != 1:
        raise ValueError("Must specify one input source type")
    if args.dataset is not None:
        ds_fn = str_to_object(args.dataset)
        if not isinstance(ds_fn, Callable):
            raise ValueError(f"{args.dataset} must be callable")
        return ds_fn(**args.dataset_kwargs)
    if args.path_tif is not None:
        if not os.path.exists(args.path_tif):
            raise ValueError(f"Path does not exists: {args.path_tif}")
        paths_tif = [args.path_tif]
        if os.path.isdir(args.path_tif):
            paths_tif = files_from_dir(args.path_tif)
        ds = TiffDataset(
            dataframe=pd.DataFrame({"path_bf": paths_tif, "path_target": None}),
            transform_signal=[norm_around_center],
            transform_target=[norm_around_center],
            col_signal="path_bf",
        )
        return ds
    raise NotImplementedError


def get_indices(args: argparse.Namespace, dataset: Any) -> List[int]:
    """Returns indices of dataset items on which to perform predictions."""
    indices = args.idx_sel
    if indices is None:
        if isinstance(dataset, FnetDataset):
            indices = dataset.df.index
        else:
            indices = list(range(len(dataset)))
    if args.n_images > 0:
        return indices[: args.n_images]
    return indices


def item_from_dataset(
    dataset: Any, idx: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns signal-target image pair from dataset.

    If the dataset is a FnetDataset, it will be indexed using 'loc'-style
    indexing.

    Parameters
    ----------
    dataset
        Object with __getitem__ implemented.
    idx
        Index of data to be retrieved from dataset.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        Signal-target data pair. Target can be None if dataset does not return
        a target for the given index.

    """
    if isinstance(dataset, FnetDataset):
        item = dataset.loc[idx]
    else:
        item = dataset[idx]
    target = None

    if isinstance(item, Tuple):
        signal = item[0]
        if len(item) > 1:
            target = item[1]
    else:
        signal = item

    # crop to the nearest size divisible by 16
    if signal.shape[2] % 16 != 0:
        signal = signal[:,:,signal.shape[2]%16//2:0-signal.shape[2]%16//2,:]
        target = target[:,:,target.shape[2]%16//2:0-target.shape[2]%16//2,:]
    if signal.shape[3] % 16 != 0:
        signal = signal[:,:,:,signal.shape[3]%16//2:0-signal.shape[3]%16//2]
        target = target[:,:,:,target.shape[3]%16//2:0-target.shape[3]%16//2]

    return (signal, target)


def save_tif(fname: str, ar: np.ndarray, path_root: str) -> str:
    """Saves a tif and returns tif save path relative to root save directory.

    Image will be stored at: 'path_root/tifs/fname'

    Parameters
    ----------
    fname
        Basename of save path.
    ar
        Array to be saved as tif.
    path_root
        Root directory of save path.

    Returns
    -------
    str
        Save path relative to root directory.

    """
    path_tif_dir = os.path.join(path_root, "tifs")
    if not os.path.exists(path_tif_dir):
        os.makedirs(path_tif_dir)
        logger.info(f"Created: {path_tif_dir}")
    path_save = os.path.join(path_tif_dir, fname)
    # change compression level to default
#     tifffile.imsave(path_save, ar, compress=2)
    tifffile.imsave(path_save, ar)
    logger.info(f"Saved: {path_save}")
    return os.path.relpath(path_save, path_root)


def parse_model(model_str: str) -> Dict:
    """Parse model definition string into dictionary."""
    model_def = {}
    parts = model_str.split(":")
    if len(parts) > 2:
        raise ValueError('Multiple ":" in specified model')
    name = os.path.basename(parts[0])
    options = []
    if len(parts) == 2:
        options.extend(parts[1].split(","))
    model_def["path"] = parts[0]
    model_def["options"] = options
    model_def["name"] = ".".join([name] + options)
    return model_def


def save_predictions_csv(
    path_csv: Path, pred_records: List[Dict], dataset: Any
) -> None:
    """Saves csv with metadata of predictions.

    Parameters
    ----------
    path_csv
        CSV save path.
    pred_records
        List of metadata for each prediction.
    dataset
        Dataset from where signal-target pairs were retrieved.

    """
    df = pd.DataFrame(pred_records).set_index("index")
    if isinstance(dataset, FnetDataset):
        # For FnetDataset, add additional metadata
        df = df.rename_axis(dataset.df.index.name).join(dataset.df, lsuffix="_pre")
    if os.path.exists(path_csv):
        df_old = pd.read_csv(path_csv)
        col_index = df_old.columns[0]  # Assumes first col is index col
        df_old = df_old.set_index(col_index)
        df = df.combine_first(df_old)
    df = df.sort_index(axis=1)
    dirname = os.path.dirname(path_csv)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info(f"Created: {dirname}")
    retry_if_oserror(df.to_csv)(path_csv)
    logger.info(f"Saved: {path_csv}")


def save_args_as_json(path_save_dir: str, args: argparse.Namespace) -> None:
    """Saves script arguments as json in save directory.

    A json is saved only if the "--json" option was not specified.

    By default, this function tries to save arguments as predict_options.json
    within the save directory. If that file already exists, appends a digit to
    uniquify the save path.

    Parameters
    ----------
    path_save_dir
        Save directory
    args
        Script arguments.

    """
    if args.json is not None:
        return
    args.__dict__.pop("json")
    path_json = os.path.join(path_save_dir, "predict_options.json")
    while os.path.exists(path_json):
        number = path_json.split(".")[-2]
        if not number.isdigit():
            number = "-1"
        number = str(int(number) + 1)
        path_json = os.path.join(
            path_save_dir, ".".join(["predict_options", number, "json"])
        )
    with open(path_json, "w") as fo:
        json.dump(vars(args), fo, indent=4, sort_keys=True)
    logger.info(f"Saved: {path_json}")


def load_from_json(args: argparse.Namespace) -> None:
    """Loads arguments from if a json is specified."""
    if args.json is None:
        return
    with args.json.open(mode="r") as fi:
        predict_options = json.load(fi)
    args.__dict__.update(predict_options)

def predict_on_zslice_tiles(model, zimage, tile_size=(512, 512), tile_step=(256, 256)):

    image = zimage[0,0,:,:]
    print(f'Stack shape:{zimage.shape}')
    print(f'Slice shape:{image.shape}')

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256))

    print(tiler.crops)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_mask_image(tile) for tile in tiler.split(image)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)),
                                                batch_size=1, pin_memory=True):
#         for x, y, tile_width, tile_height in coords_batch:
#             tile = image[y : y + tile_height, x : x + tile_width].copy()
        tiles_batch = tiles_batch.float().cuda()
        pred_batch = model(tiles_batch)

        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
#     merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1)
    merged_mask = tiler.crop_to_orignal_size(merged_mask)

    return merged_mask


def add_parser_arguments(parser) -> None:
    """Add training script arguments to parser."""
    parser.add_argument("--dataset", help="dataset name")
    parser.add_argument(
        "--dataset_kwargs", type=json.loads, default={}, help="dataset kwargs"
    )
    parser.add_argument("--gpu_ids", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--idx_sel", nargs="+", type=int, help="specify dataset indices"
    )
    parser.add_argument("--json", type=Path, help="path to prediction options json")
    parser.add_argument(
        "--metrics", default="fnet.metrics.corr_coef", help="evaluation metrics"
    )
    parser.add_argument(
        "--n_images", type=int, default=-1, help="max number of images to test"
    )
    parser.add_argument(
        "--no_prediction", action="store_true", help="set to not save predicted image"
    )
    parser.add_argument(
        "--no_signal", action="store_true", help="set to not save signal image"
    )
    parser.add_argument(
        "--no_target", action="store_true", help="set to not save target image"
    )
    parser.add_argument(
        "--path_model_dir", nargs="+", help="path(s) to model directory"
    )
    parser.add_argument(
        "--path_save_dir", default="predictions", help="path to output root directory"
    )
    parser.add_argument("--path_tif", help="path(s) to input tif(s)")


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Predicts using model."""
    if args is None:
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parser.parse_args()
    if args.json and not args.json.exists():
        save_default_predict_options(args.json)
        return
    load_from_json(args)
    metrics = [str_to_object(metric) for metric in args.metrics]
    dataset = get_dataset(args)
    entries = []
    model = None
    indices = get_indices(args, dataset)
    for count, idx in enumerate(indices, 1):
        logger.info(f"Processing: {idx:3d} ({count}/{len(indices)})")
        entry = {}
        entry["index"] = idx
        signal, target = item_from_dataset(dataset, idx)
        if not args.no_signal:
            entry["path_signal"] = save_tif(
                f"{idx}_signal.tif", signal.numpy()[0,], args.path_save_dir
            )
        if not args.no_target and target is not None:
            entry["path_target"] = save_tif(
                f"{idx}_target.tif", target.numpy()[0,], args.path_save_dir
            )
        for path_model_dir in args.path_model_dir:
            if model is None or len(args.path_model_dir) > 1:
                model_def = parse_model(path_model_dir)
                model = load_model(model_def["path"], no_optim=True)
                model.to_gpu(args.gpu_ids)
                logger.info(f'Loaded model: {model_def["name"]}')

            predictions = model.predict_piecewise(
                signal, tta=("no_tta" not in model_def["options"])
            )
#             signal = to_numpy(signal)
#             print(signal.shape)
#             print(signal.shape[1])

#             network = model.net
#             network.eval()
#             with torch.no_grad():
#                 prediction = predict_on_zslice_tiexles(network, signal)


            for i, metric in enumerate(metrics):
                evaluation = metric(target, predictions[i])
                entry[args.metrics[i] + f'.{model_def["name"]}'] = evaluation
            if not args.no_prediction:
                for idx_p, prediction in enumerate(predictions):
                    for idx_c in range(prediction.size()[0]):
                        tag = f'prediction_c{idx_c}.{model_def["name"]}'
                        pred_c = prediction.numpy()[idx_c,]
                        entry[f"path_{tag}"] = save_tif(
                            f"{idx}_{tag}_{idx_p}.tif", pred_c, args.path_save_dir
                        )
        entries.append(entry)
        save_predictions_csv(
            path_csv=os.path.join(args.path_save_dir, "predictions.csv"),
            pred_records=entries,
            dataset=dataset,
        )
    save_args_as_json(args.path_save_dir, args)


if __name__ == "__main__":
    main()
