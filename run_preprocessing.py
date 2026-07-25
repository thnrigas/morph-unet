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

import argparse

import config
from datasets.preprocessing import preprocess_data
from datasets.create_splits import create_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="parallel worker processes (default: min(cpu,16))")
    args = ap.parse_args()
    preprocess_data(root_dir=str(config.DATA_DIR), modality=config.MODALITY, channel=config.CHANNEL,
                    num_workers=args.workers)
    create_splits(output_dir=str(config.DATA_DIR), image_dir=str(config.PREPROCESSED_DIR))
    print("Preprocessing done.")
