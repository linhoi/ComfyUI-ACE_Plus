# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import sys


if sys.argv[0] == 'install.py':
    sys.path.append('.')

current_dir = os.path.dirname(__file__)
source_folder = os.path.join(current_dir, "workflow/ComfyUI-ACE_Plus")
target_folder = os.path.join(os.path.dirname(current_dir), "ComfyUI-ACE_Plus")

if not os.path.exists(target_folder):
    shutil.move(source_folder, target_folder)
    print(f"{os.path.abspath(source_folder)} copy to {os.path.abspath(target_folder)} success!")
else:
    print(f"{os.path.abspath(target_folder)} exist.")

try:
    if os.path.exists(current_dir):
        shutil.rmtree(current_dir)
    else:
        print(f"source_folder: '{current_dir}' not exist.")
except Exception as e:
    print(f"delete source_folder error: {e}")
