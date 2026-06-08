#! /usr/bin/env python

import math
import os
import argparse
import numpy as np
from create_header import BeamGenerator


def main(args):
    # load all sub-directories
    subdirs = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(args.root_dir)
        for dirname in dirnames
    ]
    for dirname in subdirs:
        if args.verbose:
            print(f"{dirname}")
        # create file names
        out_file = outfile = os.path.join(dirname, "beam.model")
        theta_file = outfile = os.path.join(dirname, "theta.npy")
        phi_file = outfile = os.path.join(dirname, "phi.npy")
        freq_file = outfile = os.path.join(dirname, "frequency.npy")
        etheta_file = outfile = os.path.join(dirname, "etheta.npy")
        ephi_file = outfile = os.path.join(dirname, "ephi.npy")
        bg = BeamGenerator(
            n0=args.order,
            beta=args.scale,
            outfile=out_file,
            preamble="",
            verbose=args.verbose,
            theta_file=theta_file,
            phi_file=phi_file,
            freq_file=freq_file,
            etheta_file=etheta_file,
            ephi_file=ephi_file,
        )
        bg.load_model()
        bg.setup_basis()
        bg.decompose_write_header()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate element beam coefficients for sagecal, to be loaded at runtime, no re-compilation needed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--scale", type=float, default=0.5, help="basis scale factor")
    parser.add_argument("--order", type=int, default=7, help="basis model order")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="",
        help="root directory under which the subdirectories with .npy files. E.g., for directories like TEST/0 TEST/1 ..., use --root_dir TEST",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print more information"
    )

    args = parser.parse_args()
    if len(args.root_dir) > 0:
        main(args)
    else:
        parser.print_help()
