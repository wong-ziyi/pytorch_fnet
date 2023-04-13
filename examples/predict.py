import os
import json
import argparse

###################################################
# Assume the user already ran download_and_train.py
###################################################

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")

args = parser.parse_args()

# Normally this would be run via command-line but this Fnet call will be updated as a python API becomes available
gpu_id = args.gpu_id

image_save_dir = "{}/images/".format(os.getcwd())
model_save_dir = "{}/model/".format(os.getcwd())

data_save_path_test = "{}/image_list_test.csv".format(os.getcwd())

dataset_kwargs = {
    "path_csv": data_save_path_test,
    "transform_signal": ["fnet.transforms.normalize"],
    "transform_target": ["fnet.transforms.normalize"],
}

command_str = (
    f"python ../fnet/cli/predict.py "
    f"--path_model_dir {model_save_dir} "
    f"--dataset fnet.data.MultiChTiffDataset "
    f"--dataset_kwargs \'{json.dumps(dataset_kwargs)}\' "
    f"--gpu_ids {gpu_id}"
)

print(command_str)
os.system(command_str)
