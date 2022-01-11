#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 ITMO University (Aleksandr Laptev)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model CTC alignment script."""

import configargparse
import logging
import os
import random
import subprocess
import sys

import numpy as np

from espnet.asr.pytorch_backend.recog import calc_logits

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`",
    )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for logit inference",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    # task related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--result-mat",
        type=str,
        required=True,
        help="Filename of result logit data (kaldi ark)",
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args, extra = parser.parse_known_args(args)

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # prevent memory leak when using cpu
    os.environ["LRU_CACHE_CAPACITY"] = "1"
    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

        # trying to recover from the run.pl's CUDA_VISIBLE_DEVICES shifted allocation
        ngpu_command = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
        sys_ngpu = subprocess.Popen(
            [ngpu_command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
        sys_ngpu = int(sys_ngpu[0].decode("utf-8").split("\n")[0])
        cvd = int(cvd)
        if cvd == sys_ngpu:
            logging.warning(
                """CUDA_VISIBLE_DEVICES is set to a non-existing device.
 Overwritten as '0'."""
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif cvd > sys_ngpu:
            logging.error("CUDA_VISIBLE_DEVICES is set to a non-existing device.")
            sys.exit(1)
    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # recog
    logging.info("backend = " + args.backend)
    if args.backend == "pytorch":
        calc_logits(args)
    else:
        raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
