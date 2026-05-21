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
import os

from datasets.preprocessing import preprocess_data
from datasets.create_splits import create_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/Task04_Hippocampus")
    args = p.parse_args()

    preprocess_data(root_dir=args.root)
    create_splits(output_dir=args.root, image_dir=os.path.join(args.root, "preprocessed"))
    print("Preprocessing done.")
