# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
from loguru import logger

import modules.args
import modules.lib

if __name__ == "__main__":

    logger.info("Baseline `mxl-inventory`")

    args = modules.args.global_args("Baseline `mxl-inventory`")
    all_path_data = modules.lib.get_data_paths(args)

    for k in all_path_data:
        i = all_path_data[k]["seed_offset"]
        logger.info(f"Starting data set {i:03d}")
        os.system(f"Rscript baselines/mxl.R -s {i} -c {args.c}")
