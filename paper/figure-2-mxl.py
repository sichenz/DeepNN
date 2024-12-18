# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
from loguru import logger

if __name__ == "__main__":
    logger.info("Run MXL for Figure 2")
    os.system(f"Rscript paper/figure-2-mxl.R")
