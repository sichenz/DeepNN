# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import json
import yaml
import shutil
import glob
import re
import hashlib
import pickle
import pathlib
import scipy

import numpy as np
import pandas as pd

import sklearn.metrics

from loguru import logger


# read data from and write data to YAML files
def read_yaml(f):
    with open(f, "r") as con:
        out = yaml.safe_load(con)
    return out


def write_yaml(x, f):
    with open(f, "w") as con:
        yaml.safe_dump(x, con)


# read data from and write data to PICKLE files
def dump_pickle(x, f):
    with open(f, "wb") as con:
        pickle.dump(x, con, protocol=4)


def load_pickle(f):
    with open(f, "rb") as con:
        return pickle.load(con)


def load_gzip_pickle(f):
    assert os.path.isfile(f)
    f_tmp = f"{os.path.dirname(f)}/askdfjasldfjasldkfj.pickle.gz"
    f_tmp_2 = f"{os.path.dirname(f)}/askdfjasldfjasldkfj.pickle"
    assert not os.path.isfile(f_tmp)
    shutil.copy(f, f_tmp)
    os.system(f"gunzip {f_tmp}")
    out = load_pickle(f_tmp_2)
    os.remove(f_tmp_2)
    return out


# md5sum
def md5_dict(x):
    return hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()


#  build manifest file that contains md5 keys for all files in a given folder
def md5(f):
    hash_md5 = hashlib.md5()
    with open(f, "rb") as con:
        for chunk in iter(lambda: con.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def build_manifest(path, pattern="*", exclude=".*MANIFEST.yaml$", file="MANIFEST.yaml"):
    files = glob.glob(f"{path}/{pattern}")
    if exclude is not None:
        files = [f for f in files if not re.match(exclude, f)]
    files = sorted(files)
    manifest = {}
    for f in files:
        manifest[os.path.basename(f)] = md5(f)
    w


# build dict containing all data paths (useful for bootstrap)
def get_data_paths(args, create_data=False, rerun=False):
    # parse args
    x = read_yaml(args.c)
    if args.t is not None:
        torch_config = read_yaml(args.t)
    else:
        torch_config = {
            "model": "",
            "experiment_suffix": "",
        }
    label = args.l

    # extract all path from config
    path_data_raw = os.path.expandvars(
        (x if not create_data else x["simulated_data"][args.d])["path_data"]
    )
    if label is None:
        all_paths = {
            path_data_raw: {
                "path_data": path_data_raw,
                "model": torch_config["model"],
                "suffix": torch_config["experiment_suffix"],
                "seed_offset": 0,
                "label": os.path.basename(path_data_raw),
                **x["bootstrap"]["params"],
            }
        }
    elif "n_data_sets" in x[label]:
        # bootstrap
        all_paths = {
            f"{path_data_raw}_{s:03d}": {
                "path_data": f"{path_data_raw}_{s:03d}",
                "model": torch_config["model"],
                "suffix": torch_config["experiment_suffix"],
                "seed_offset": s,
                **x["bootstrap"]["params"],
            }
            for s in range(x["bootstrap"]["n_data_sets"])
        }
    else:
        raise Exception("unknown `label`")
    model_folder = "{model}{experiment_suffix}".format(**torch_config)

    # return all paths if rerun is required
    if rerun:
        paths = all_paths
    # if rerun is NOT required, only run incomplete or inactive paths
    else:
        paths = {}
        for p in all_paths:
            file_experiment = f"{p}/experiment.yaml"
            if not os.path.isfile(file_experiment):
                paths[p] = all_paths[p]
            else:
                experiment_p = read_yaml(f"{p}/experiment.yaml")
                if "status" not in experiment_p:
                    paths[p] = all_paths[p]
                if experiment_p["status"] == "":
                    paths[p] = all_paths[p]
    return paths


# create empty file
def touch(f):
    pathlib.Path(f).touch()


# check whether results exist
def check_state(f_in, f_in_size, f_out, path_data):
    state_in = (
        not os.path.isfile(f_in) or os.stat(f_in).st_size / (1024 ** 2) < f_in_size
    )
    if state_in:
        logger.warning(f"`{f_in}` does not exist, skipping {path_data}...")
    state_out = os.path.isfile(f_out)
    if state_out:
        logger.warning(f"`{f_out}` exists, skipping {path_data}...")
    return state_in or state_out


# check whether bootstrap iteration should run in this pipeline
def check_bootstrap_iter(x, y, z):
    if y is not None and z is not None:
        if (x % y) != z:
            return True
        else:
            return False
    else:
        return False


# METRICS
def get_hitrate(x, n):
    # extract values from input data
    xys = x[["x", "y"]].values
    js = x["j"].values
    cs = x["c"].values

    # compute Euclidean distances between all products
    distance_in_plot = scipy.spatial.distance.cdist(xys, xys)

    # build distance DataFrame
    distance_df = pd.DataFrame(
        {
            "j": np.repeat(js, len(js)),
            "c": np.repeat(cs, len(cs)),
            "j2": np.tile(js, len(js)),
            "c2": np.tile(cs, len(cs)),
            "d": distance_in_plot.flatten(),
        }
    )
    assert distance_df[["j", "c"]].drop_duplicates().shape[0] == len(js)
    assert distance_df[["j2", "c2"]].drop_duplicates().shape[0] == len(js)

    # remove cases where j == j2 ("self-distance", which is 0)
    distance_df = distance_df[distance_df["j"] != distance_df["j2"]]

    # prune data to nearest neighbors
    distance_df = distance_df.sort_values("d")
    distance_df["rank_d"] = distance_df.groupby("j").cumcount()
    nn = distance_df[distance_df["rank_d"] < n]

    # calculate score:
    # fraction of NN for which category is identical to reference product
    score = float(sum(nn["c"] == nn["c2"])) / nn.shape[0]
    return score


def score_ami(x):
    ami = sklearn.metrics.adjusted_mutual_info_score(
        labels_true=x.c_kmeans.values,
        labels_pred=x.c.values,
        average_method="arithmetic",
    )
    return round(float(ami), 4)


def score_nn(x, N):
    nn = get_hitrate(x, N)
    return round(float(nn), 4)


def score_sil(x):
    sil = sklearn.metrics.silhouette_score(
        X=x[["x", "y"]].values, labels=x.c_kmeans.values
    )
    return round(float(sil), 4)
