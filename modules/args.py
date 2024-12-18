# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import argparse


def global_args(description, def_t=None):

    # args
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-a", action="store_true", default=False, help="run all configs."
    )
    parser.add_argument(
        "-c",
        type=str,
        default="configs/config.yaml",
        help="config file.",
    )
    parser.add_argument(
        "-d",
        type=str,
        default="config-05",
        help="data set configuration.",
    )
    parser.add_argument(
        "-e",
        type=int,
        default=99,
        help="prediction epoch.",
    )
    parser.add_argument(
        "-emb",
        type=str,
        default="w2v",
        help="pre-trained embedding.",
    )
    parser.add_argument("-f", type=str, help="file location (optional).")
    parser.add_argument(
        "-l",
        type=str,
        default="bootstrap",
        help="label of suffix set in config (optional).",
    )
    parser.add_argument(
        "-m",
        type=str,
        default="loss",
        help="metric.",
    )
    parser.add_argument("-p", action="store_false", help="pickle data loader.")
    parser.add_argument(
        "-r", action="store_true", default=False, help="force model rerun."
    )
    parser.add_argument(
        "-s", action="store_true", default=False, help="share results on dropbox."
    )
    parser.add_argument(
        "-t",
        type=str,
        default=def_t,
        help="torch config file.",
    )
    parser.add_argument(
        "-uemb",
        type=str,
        default="wm",
        help="user embedding type.",
    )
    parser.add_argument(
        "-y",
        type=int,
        help="divisor for modulo",
    )
    parser.add_argument(
        "-z",
        type=int,
        help="remainder for modulo.",
    )
    return parser.parse_args()
