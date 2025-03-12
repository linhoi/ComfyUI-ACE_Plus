# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import glob
import importlib
import io
import os
import sys

from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from examples.examples import fft_examples as all_examples
from inference.registry import INFERENCES
fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


def run_one_case(pipe,
                input_image = None,
                input_mask = None,
                input_reference_image = None,
                save_path = "examples/output/example.png",
                instruction = "",
                output_h = 1024,
                output_w = 1024,
                seed = -1,
                sample_steps = None,
                guide_scale = None,
                repainting_scale = None,
                use_change=True,
                keep_pixels=True,
                keep_pixels_rate=0.8,
                **kwargs):
    if input_image is not None:
        input_image = Image.open(io.BytesIO(FS.get_object(input_image)))
        input_image = pillow_convert(input_image, "RGB")
    if input_mask is not None:
        input_mask = Image.open(io.BytesIO(FS.get_object(input_mask)))
        input_mask = pillow_convert(input_mask, "L")
    if input_reference_image is not None:
        input_reference_image = Image.open(io.BytesIO(FS.get_object(input_reference_image)))
        input_reference_image = pillow_convert(input_reference_image, "RGB")
    print(repainting_scale)
    image, _, _, _, seed = pipe(
        reference_image=input_reference_image,
        edit_image=input_image,
        edit_mask=input_mask,
        prompt=instruction,
        output_height=output_h,
        output_width=output_w,
        sampler='flow_euler',
        sample_steps=sample_steps or pipe.input.get("sample_steps", 28),
        guide_scale=guide_scale or pipe.input.get("guide_scale", 50),
        seed=seed,
        repainting_scale=repainting_scale,
        use_change=use_change,
        keep_pixels=keep_pixels,
        keep_pixels_rate=keep_pixels_rate
    )
    with FS.put_to(save_path) as local_path:
        image.save(local_path)
    return local_path, seed


def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--instruction',
                        dest='instruction',
                        help='The instruction for editing or generating!',
                        default="")
    parser.add_argument('--output_h',
                        dest='output_h',
                        help='The height of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--output_w',
                        dest='output_w',
                        help='The width of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--input_reference_image',
                        dest='input_reference_image',
                        help='The input reference image!',
                        default=None
                        )
    parser.add_argument('--input_image',
                        dest='input_image',
                        help='The input image!',
                        default=None
                        )
    parser.add_argument('--input_mask',
                        dest='input_mask',
                        help='The input mask!',
                        default=None
                        )
    parser.add_argument('--save_path',
                        dest='save_path',
                        help='The save path for output image!',
                        default='examples/output_images/output.png'
                        )
    parser.add_argument('--seed',
                        dest='seed',
                        help='The seed for generation!',
                        type=int,
                        default=-1)

    parser.add_argument('--step',
                        dest='step',
                        help='The sample step for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--guide_scale',
                        dest='guide_scale',
                        help='The guide scale for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--repainting_scale',
                        dest='repainting_scale',
                        help='The repainting scale for content filling generation!',
                        type=int,
                        default=None)

    cfg = Config(load=True, parser_ins=parser)
    model_cfg = Config(load=True, cfg_file="config/ace_plus_fft.yaml")
    pipe = INFERENCES.build(model_cfg)


    if cfg.args.instruction == "" and cfg.args.input_image is None and cfg.args.input_reference_image is None:
        params = {
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale
        }
        # run examples

        for example in all_examples:
            example.update(params)
            local_path, seed = run_one_case(pipe, **example)

    else:
        params = {
            "input_image": cfg.args.input_image,
            "input_mask": cfg.args.input_mask,
            "input_reference_image": cfg.args.input_reference_image,
            "save_path": cfg.args.save_path,
            "instruction": cfg.args.instruction,
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale,
            "repainting_scale": cfg.args.repainting_scale,
        }
        local_path, seed = run_one_case(pipe, **params)
        print(local_path, seed)

if __name__ == '__main__':
    run()

