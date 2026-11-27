#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# LAYOUT (environment, see config.py) picks the array layout and the output folder:
#   python3 run_preprocessing.py                -> data/<task>/preprocessed/        (original, x y z)
#   LAYOUT=axial python3 run_preprocessing.py   -> data/<task>/preprocessed_axial/  (axial slices)
#
import argparse
import pickle

import config
from datasets.preprocessing import preprocess_data
from datasets.create_splits import create_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="parallel worker processes (default: min(cpu,16))")
    args = ap.parse_args()
    preprocess_data(root_dir=str(config.DATA_DIR), modality=config.MODALITY, channel=config.CHANNEL,
                    num_workers=args.workers, layout=config.LAYOUT, output_dir=str(config.PREPROCESSED_DIR))
    # every result is tied to the folds, so an existing splits.pkl is never rewritten, only checked: the
    # split is deterministic in the sorted case names, so the same cases must give the same folds
    existed = config.SPLITS_FILE.exists()
    splits = create_splits(output_dir=str(config.DATA_DIR), image_dir=str(config.PREPROCESSED_DIR),
                           write=not existed)
    if existed:
        with open(config.SPLITS_FILE, "rb") as f:
            if pickle.load(f) != splits:
                raise SystemExit(f"!! the folds recomputed from {config.PREPROCESSED_DIR} differ from "
                                 f"{config.SPLITS_FILE} (another case set?); splits.pkl was NOT changed")
        print(f"splits.pkl unchanged: the folds recomputed from {config.PREPROCESSED_DIR} are identical")
    else:
        print(f"splits.pkl written: {config.SPLITS_FILE}")
    print("Preprocessing done.")
