# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import csv
import glob
import os
import sys
import threading
import time

import gradio as gr
import numpy as np
import torch, importlib
from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

inference_dict = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference
}

fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


csv.field_size_limit(sys.maxsize)
refresh_sty = '\U0001f504'  # 🔄
clear_sty = '\U0001f5d1'  # 🗑️
upload_sty = '\U0001f5bc'  # 🖼️
sync_sty = '\U0001f4be'  # 💾
chat_sty = '\U0001F4AC'  # 💬
video_sty = '\U0001f3a5'  # 🎥

lock = threading.Lock()
class DemoUI(object):
    def __init__(self,
                 infer_dir = "./config/ace_plus_diffusers_infer.yaml",
                 model_list='./models/model_zoo.yaml'
                 ):
        self.model_yamls = [infer_dir]
        self.model_choices = dict()
        self.default_model_name = ''
        for i in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=i)
            model_name = model_cfg.NAME
            if model_cfg.IS_DEFAULT: self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
        print('Models: ', self.model_choices.keys())
        assert len(self.model_choices) > 0
        if self.default_model_name == "": self.default_model_name = list(self.model_choices.keys())[0]
        self.model_name = self.default_model_name
        pipe_cfg = self.model_choices[self.default_model_name]
        infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE")
        self.pipe = inference_dict[infer_name]()
        self.pipe.init_from_cfg(pipe_cfg)

        # choose different model
        self.task_model_cfg = Config(load=True, cfg_file=model_list)
        self.task_model = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]
        for task_name, task_model in self.task_model_cfg.MODEL.items():
            self.task_model[task_name.lower()] = task_model
            self.task_model_list.append(task_name.lower())
            for preprocessor in task_model.get("PREPROCESSOR", []):
                if preprocessor["TYPE"] in self.edit_type_dict:
                    continue
                preprocessor["REPAINTING_SCALE"] = task_model.get("REPAINTING_SCALE", 1.0)
                self.edit_type_dict[preprocessor["TYPE"]] = preprocessor
        self.max_msgs = 20
        # reformat examples
        self.all_examples = [
            [
             one_example["task_type"], one_example["edit_type"], one_example["instruction"],
             one_example["input_reference_image"], one_example["input_image"],
             one_example["input_mask"], one_example["output_h"],
             one_example["output_w"], one_example["seed"]
             ]
            for one_example in all_examples
        ]

    def construct_edit_image(self, edit_image, edit_mask):
        if edit_image is not None and edit_mask is not None:
            edit_image_rgb = pillow_convert(edit_image, "RGB")
            edit_image_rgba = pillow_convert(edit_image, "RGBA")
            edit_mask = pillow_convert(edit_mask, "L")

            arr1 = np.array(edit_image_rgb)
            arr2 = np.array(edit_mask)[:, :, np.newaxis]
            result_array = np.concatenate((arr1, arr2), axis=2)
            layer = Image.fromarray(result_array)

            ret_data = {
                "background": edit_image_rgba,
                "composite": edit_image_rgba,
                "layers": [layer]
            }
            return ret_data
        else:
            return None




    def create_ui(self):
        with gr.Row(equal_height=True, visible=True):
            with gr.Column(scale=2):
                self.gallery_image = gr.Image(
                    height=600,
                    interactive=False,
                    type='pil',
                    elem_id='Reference_image'
                )
            with gr.Column(scale=1, visible=True) as self.edit_preprocess_panel:
                with gr.Row():
                    with gr.Accordion(label='Related Input Image', open=False):
                        self.edit_preprocess_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type='pil',
                            elem_id='preprocess_image'
                        )

                        self.edit_preprocess_mask_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type='pil',
                            elem_id='preprocess_image_mask'
                        )
                with gr.Row():
                    instruction = """
                               **Instruction**:
                               1. Please choose the Task Type based on the scenario of the generation task. We provide three types of generation capabilities: Portrait ID Preservation Generation(portrait), 
                                Object ID Preservation Generation(subject), and Local Controlled Generation(local editing), which can be selected from the task dropdown menu.
                               2. When uploading images in the Reference Image section, the generated image will reference the ID information of that image. Please ensure that the ID information is clear. 
                                In the Edit Image section, the uploaded image will maintain its structural and content information, and you must draw a mask area to specify the region to be regenerated.
                               3. When the task type is local editing, there are various editing types to choose from. Users can select different information preserving dimensions, such as edge information, 
                                color information, and more. The pre-processing information can be viewed in the 'related input image' tab.
                            """
                    self.instruction = gr.Markdown(value=instruction)
        with gr.Row():
            self.model_name_dd = gr.Dropdown(
                choices=self.model_choices,
                value=self.default_model_name,
                label='Model Version')
            self.task_type = gr.Dropdown(choices=self.task_model_list,
                                         interactive=True,
                                         value=self.task_model_list[0],
                                         label='Task Type')
            self.edit_type = gr.Dropdown(choices=self.edit_type_list,
                                         interactive=True,
                                         value=self.edit_type_list[0],
                                         label='Edit Type')
        with gr.Row():
            self.generation_info_preview = gr.Markdown(
                label='System Log.',
                show_label=True)
        with gr.Row(variant='panel',
                    equal_height=True,
                    show_progress=False):
            with gr.Column(scale=10, min_width=500):
                self.text = gr.Textbox(
                    placeholder='Input "@" find history of image',
                    label='Instruction',
                    container=False,
                    lines = 1)
            with gr.Column(scale=2, min_width=100):
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        self.chat_btn = gr.Button(value='Generate', variant = "primary")

        with gr.Accordion(label='Advance', open=True):
            with gr.Row(visible=True):
                with gr.Column():
                    self.reference_image = gr.Image(
                        height=1000,
                        interactive=True,
                        image_mode='RGB',
                        type='pil',
                        label='Reference Image',
                        elem_id='reference_image'
                    )
                with gr.Column():
                    self.edit_image = gr.ImageMask(
                        height=1000,
                        interactive=True,
                        value=None,
                        sources=['upload'],
                        type='pil',
                        layers=False,
                        label='Edit Image',
                        elem_id='image_editor',
                        show_fullscreen_button=True,
                        format="png"
                    )

            with gr.Row():
                self.step = gr.Slider(minimum=1,
                                      maximum=1000,
                                      value=self.pipe.input.get("sample_steps", 20),
                                      visible=self.pipe.input.get("sample_steps", None) is not None,
                                      label='Sample Step')
                self.cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=100.0,
                    value=self.pipe.input.get("guide_scale", 4.5),
                    visible=self.pipe.input.get("guide_scale", None) is not None,
                    label='Guidance Scale')
                self.seed = gr.Slider(minimum=-1,
                                      maximum=10000000,
                                      value=-1,
                                      label='Seed')
                self.output_height = gr.Slider(
                    minimum=256,
                    maximum=1440,
                    value=self.pipe.input.get("output_height", 1024),
                    visible=self.pipe.input.get("output_height", None) is not None,
                    label='Output Height')
                self.output_width = gr.Slider(
                    minimum=256,
                    maximum=1440,
                    value=self.pipe.input.get("output_width", 1024),
                    visible=self.pipe.input.get("output_width", None) is not None,
                    label='Output Width')

                self.repainting_scale = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=self.pipe.input.get("repainting_scale", 1.0),
                    visible=True,
                    label='Repainting Scale')
            with gr.Row():
                self.eg = gr.Column(visible=True)



    def set_callbacks(self, *args, **kwargs):
        ########################################
        def change_model(model_name):
            if model_name not in self.model_choices:
                gr.Info('The provided model name is not a valid choice!')
                return model_name, gr.update(), gr.update()

            if model_name != self.model_name:
                lock.acquire()
                del self.pipe
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                pipe_cfg = self.model_choices[model_name]
                infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE")
                self.pipe = inference_dict[infer_name]()
                self.pipe.init_from_cfg(pipe_cfg)
                self.model_name = model_name
                lock.release()

            return (model_name, gr.update(),
                    gr.Slider(
                              value=self.pipe.input.get("sample_steps", 20),
                              visible=self.pipe.input.get("sample_steps", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("guide_scale", 4.5),
                        visible=self.pipe.input.get("guide_scale", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("output_height", 1024),
                        visible=self.pipe.input.get("output_height", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("output_width", 1024),
                        visible=self.pipe.input.get("output_width", None) is not None),
                    gr.Slider(value=self.pipe.input.get("repainting_scale", 1.0))
                    )

        self.model_name_dd.change(
            change_model,
            inputs=[self.model_name_dd],
            outputs=[
                self.model_name_dd, self.text,
                self.step,
                self.cfg_scale,
                self.output_height,
                self.output_width,
                self.repainting_scale])

        def change_task_type(task_type):
            task_info = self.task_model[task_type]
            edit_type_list = [self.edit_type_list[0]]
            for preprocessor in task_info.get("PREPROCESSOR", []):
                preprocessor["REPAINTING_SCALE"] = task_info.get("REPAINTING_SCALE", 1.0)
                self.edit_type_dict[preprocessor["TYPE"]] = preprocessor
                edit_type_list.append(preprocessor["TYPE"])

            return gr.update(choices=edit_type_list, value=edit_type_list[0])

        self.task_type.change(change_task_type, inputs=[self.task_type], outputs=[self.edit_type])

        def change_edit_type(edit_type):
            edit_info = self.edit_type_dict[edit_type]
            edit_info = edit_info or {}
            repainting_scale = edit_info.get("REPAINTING_SCALE", 1.0)
            if edit_type == self.edit_type_list[0]:
                return  gr.Slider(value=1.0)
            else:
                return  gr.Slider(
                    value=repainting_scale)

        self.edit_type.change(change_edit_type, inputs=[self.edit_type], outputs=[self.repainting_scale])

        def preprocess_input(ref_image, edit_image_dict, preprocess = None):
            err_msg = ""
            is_suc = True
            if ref_image is not None:
                ref_image = pillow_convert(ref_image, "RGB")

            if edit_image_dict is None:
                edit_image = None
                edit_mask = None
            else:
                edit_image = edit_image_dict["background"]
                edit_mask = np.array(edit_image_dict["layers"][0])[:, :, 3]
                if np.sum(np.array(edit_image)) < 1:
                    edit_image = None
                    edit_mask = None
                elif np.sum(np.array(edit_mask)) < 1:
                    err_msg = "You must draw the repainting area for the edited image."
                    return None, None, None, False, err_msg
                else:
                    edit_image = pillow_convert(edit_image, "RGB")
                    edit_mask = Image.fromarray(edit_mask).convert('L')
            if ref_image is None and edit_image is None:
                err_msg = "Please provide the reference image or edited image."
                return None, None, None, False, err_msg
            return edit_image, edit_mask, ref_image, is_suc, err_msg

        def run_chat(
                     prompt,
                     ref_image,
                     edit_image,
                     task_type,
                     edit_type,
                     cfg_scale,
                     step,
                     seed,
                     output_h,
                     output_w,
                     repainting_scale,
                progress=gr.Progress(track_tqdm=True)
        ):
            model_path = self.task_model[task_type]["MODEL_PATH"]
            edit_info = self.edit_type_dict[edit_type]

            if task_type in ["portrait", "subject"] and ref_image is None:
                err_msg = "<mark>Please provide the reference image.</mark>"
                return (gr.Image(), gr.Column(visible=True),
                        gr.Image(),
                        gr.Image(),
                        gr.Text(value=err_msg))

            pre_edit_image, pre_edit_mask, pre_ref_image, is_suc, err_msg = preprocess_input(ref_image, edit_image)
            if not is_suc:
                err_msg = f"<mark>{err_msg}</mark>"
                return (gr.Image(), gr.Column(visible=True),
                        gr.Image(),
                        gr.Image(),
                        gr.Text(value=err_msg))
            pre_edit_image = edit_preprocess(edit_info, we.device_id, pre_edit_image, pre_edit_mask)
            # edit_image["background"] = pre_edit_image
            st = time.time()
            image, seed = self.pipe(
                reference_image=pre_ref_image,
                edit_image=pre_edit_image,
                edit_mask=pre_edit_mask,
                prompt=prompt,
                output_height=output_h,
                output_width=output_w,
                sampler='flow_euler',
                sample_steps=step,
                guide_scale=cfg_scale,
                seed=seed,
                repainting_scale=repainting_scale,
                lora_path = model_path
            )
            et = time.time()
            msg = f"prompt: {prompt}; seed: {seed}; cost time: {et - st}s; repaiting scale: {repainting_scale}"

            return (gr.Image(value=image), gr.Column(visible=True),
                    gr.Image(value=pre_edit_image if pre_edit_image is not None else pre_ref_image),
                    gr.Image(value=pre_edit_mask if pre_edit_mask is not None else None),
                    gr.Text(value=msg))

        chat_inputs = [
            self.reference_image,
            self.edit_image,
            self.task_type,
            self.edit_type,
            self.cfg_scale,
            self.step,
            self.seed,
            self.output_height,
            self.output_width,
            self.repainting_scale
        ]

        chat_outputs = [
           self.gallery_image, self.edit_preprocess_panel, self.edit_preprocess_preview,
            self.edit_preprocess_mask_preview, self.generation_info_preview
        ]

        self.chat_btn.click(run_chat,
                            inputs=[self.text] + chat_inputs,
                            outputs=chat_outputs,
                            queue=True)

        self.text.submit(run_chat,
                         inputs=[self.text] + chat_inputs,
                         outputs=chat_outputs,
                         queue=True)

        def run_example(task_type, edit_type, prompt, ref_image, edit_image, edit_mask,
                        output_h, output_w, seed, progress=gr.Progress(track_tqdm=True)):
            model_path = self.task_model[task_type]["MODEL_PATH"]

            step = self.pipe.input.get("sample_steps", 20)
            cfg_scale = self.pipe.input.get("guide_scale", 20)

            edit_info = self.edit_type_dict[edit_type]

            edit_image = self.construct_edit_image(edit_image, edit_mask)

            pre_edit_image, pre_edit_mask, pre_ref_image, _, _ = preprocess_input(ref_image, edit_image)
            pre_edit_image = edit_preprocess(edit_info, we.device_id, pre_edit_image, pre_edit_mask)
            edit_info = edit_info or {}
            repainting_scale = edit_info.get("REPAINTING_SCALE", 1.0)
            st = time.time()
            image, seed = self.pipe(
                reference_image=pre_ref_image,
                edit_image=pre_edit_image,
                edit_mask=pre_edit_mask,
                prompt=prompt,
                output_height=output_h,
                output_width=output_w,
                sampler='flow_euler',
                sample_steps=step,
                guide_scale=cfg_scale,
                seed=seed,
                repainting_scale=repainting_scale,
                lora_path=model_path
            )
            et = time.time()
            msg = f"prompt: {prompt}; seed: {seed}; cost time: {et - st}s; repaiting scale: {repainting_scale}"
            if pre_edit_image is not None:
                ret_image = Image.composite(Image.new("RGB", pre_edit_image.size, (0, 0, 0)), pre_edit_image,  pre_edit_mask)
            else:
                ret_image = None
            return (gr.Image(value=image), gr.Column(visible=True),
                    gr.Image(value=pre_edit_image if pre_edit_image is not None else pre_ref_image),
                    gr.Image(value=pre_edit_mask if pre_edit_mask is not None else None),
                    gr.Text(value=msg),
                    gr.update(value=ret_image))

        with self.eg:
            self.example_edit_image = gr.Image(label='Edit Image',
                                          type='pil',
                                          image_mode='RGB',
                                          visible=False)
            self.example_edit_mask = gr.Image(label='Edit Image Mask',
                                         type='pil',
                                         image_mode='L',
                                         visible=False)

            self.examples = gr.Examples(
                fn=run_example,
                examples=self.all_examples,
                inputs=[
                    self.task_type, self.edit_type, self.text, self.reference_image, self.example_edit_image,
                    self.example_edit_mask, self.output_height, self.output_width, self.seed
                ],
                outputs=[self.gallery_image, self.edit_preprocess_panel, self.edit_preprocess_preview,
                         self.edit_preprocess_mask_preview, self.generation_info_preview, self.edit_image],
                examples_per_page=6,
                cache_examples=False,
                run_on_click=True)


def run_gr(cfg):
    with gr.Blocks() as demo:
        chatbot = DemoUI()
        chatbot.create_ui()
        chatbot.set_callbacks()
        demo.launch(server_name='0.0.0.0',
                    server_port=cfg.args.server_port,
                    root_path=cfg.args.root_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--server_port',
                        dest='server_port',
                        help='',
                        type=int,
                        default=2345)
    parser.add_argument('--root_path', dest='root_path', help='', default='')
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
