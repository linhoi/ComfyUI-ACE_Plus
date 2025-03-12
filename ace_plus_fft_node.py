# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import folder_paths, os
from comfy.supported_models import FluxInpaint, models
from nodes import UNETLoader
try:
    from scepter.modules.utils.file_system import FS
    from scepter.modules.annotator.registry import ANNOTATORS
    from scepter.modules.utils.config import Config

    fs_list = [
        Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./"}, load=False),
        Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./"}, load=False),
        Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./"}, load=False),
        Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./"}, load=False)
    ]

    for one_fs in fs_list:
        FS.init_fs_client(one_fs)
    SCEPTER = True
except:
    SCEPTER = False

class ACEPlus(FluxInpaint):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
        "in_channels": 112,
    }


class ACEPlusFFTLoader(UNETLoader):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                             "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "ComfyUI-ACE_Plus"

    def load_unet(self, unet_name, weight_dtype):
        models.append(ACEPlus)
        return super().load_unet(unet_name, weight_dtype)


import torch
import node_helpers


class ACEPlusFFTConditioning:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "ucpixels": ("IMAGE", ),
                             "cpixels": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "noise_mask": ("BOOLEAN", {"default": True, "tooltip": "Add a noise mask to the latent "
                                                                                    "so sampling will only happen "
                                                                                    "within the mask. Might improve "
                                                                                    "results or completely break "
                                                                                    "things depending on the model."}),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "ComfyUI-ACE_Plus"

    def encode(self,
               positive,
               negative,
               vae,
               ucpixels,
               cpixels,
               mask,
               noise_mask=True):
        x = (ucpixels.shape[1] // 8) * 8
        y = (ucpixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(ucpixels.shape[1], ucpixels.shape[2]), mode="bilinear")

        orig_pixels = ucpixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        orig_c_pixels = cpixels
        c_pixels = orig_c_pixels.clone()
        if orig_c_pixels.shape[1] != x or orig_c_pixels.shape[2] != y:
            x_offset = (orig_c_pixels.shape[1] % 8) // 2
            y_offset = (orig_c_pixels.shape[2] % 8) // 2
            c_pixels = orig_c_pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)
        c_concat_latent = vae.encode(c_pixels)

        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {
                    "concat_latent_image": torch.cat([concat_latent, c_concat_latent], dim=1),
                    "concat_mask": mask})
            out.append(c)

        return (out[0], out[1], out_latent)


import torch
import math
import os
import yaml
import torchvision.transforms as T
import numpy as np
from PIL import Image


class AcePlusFFTProcessor:
    def __init__(self,
                 max_aspect_ratio=4,
                 d=16,
                 max_seq_len=1024):
        self.max_aspect_ratio = max_aspect_ratio
        self.max_seq_len = max_seq_len
        self.d = d
        self.processor_cfg = self.load_yaml(os.path.join('custom_nodes/ComfyUI-ACE_Plus/',
                                                         'config/ace_plus_fft_processor.yaml'))
        self.task_list = {}
        for task in self.processor_cfg['PREPROCESSOR']:
            self.task_list[task['TYPE']] = task
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])
        ])

    CATEGORY = 'ComfyUI-ACE_Plus'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'use_reference': ('BOOLEAN', {'default': True}),
                'height': ('INT', {
                    'default': 1024,
                    'min': 256,
                    'max': 1436,
                    'step': 16
                }),
                'width': ('INT', {
                    'default': 1024,
                    'min': 256,
                    'max': 1436,
                    'step': 16
                }),
                'task_type': (list(s().task_list.keys()),),
                'keep_pixels_rate': ('FLOAT', {
                    'default': 0.8,
                    'min': 0,
                    'max': 1,
                    'step': 0.01
                }),
                'max_seq_length': ('INT', {
                    'default': 3072,
                    'min': 1024,
                    'max': 5120,
                    'step': 0.01
                }),
            },
            'optional': {
                'reference_image': ('IMAGE',),
                'edit_image': ('IMAGE',),
                'edit_mask': ('MASK',),

            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK',  'INT', 'INT', 'INT')
    RETURN_NAMES = ('UC_IMAGE', 'C_IMAGE', 'MASK',  'OUT_H', 'OUT_W', 'SLICE_W')
    FUNCTION = 'preprocess'

    def load_yaml(self, cfg_file):
        with open(cfg_file, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg

    def image_check(self, image):
        if image is None:
            return image
        # preprocess
        H, W = image.shape[1: 3]
        image = image.permute(0, 3, 1, 2)
        if H / W > self.max_aspect_ratio:
            image[0] = T.CenterCrop([int(self.max_aspect_ratio * W), W])(image[0])
        elif W / H > self.max_aspect_ratio:
            image[0] = T.CenterCrop([H, int(self.max_aspect_ratio * H)])(image[0])
        return image[0]

    def trans_pil_tensor(self, pil_image):
        transform = T.Compose([
            T.ToTensor()
        ])
        tensor_image = transform(pil_image)
        return tensor_image

    def edit_preprocess(self, processor, device, edit_image, edit_mask):

        if edit_image is None or processor is None:
            return edit_image
        if not SCEPTER:
            raise ImportError(f'Please install scepter to use edit processor {processor} by '
                              f'runing "pip install scepter" in the conda env')
        processor = Config(cfg_dict=processor, load=False)
        processor = ANNOTATORS.build(processor).to(device)
        edit_image = Image.fromarray(np.array(edit_image[0] * 255).astype(np.uint8)).convert('RGB')
        new_edit_image = processor(np.asarray(edit_image))

        del processor
        new_edit_image = Image.fromarray(new_edit_image)
        if edit_mask is not None:
            edit_mask = np.where(edit_mask > 0.5, 1, 0) * 255
            edit_mask = Image.fromarray(np.array(edit_mask[0]).astype(np.uint8)).convert('L')

        if new_edit_image.size != edit_image.size:
            edit_image = T.Resize((edit_image.size[1], edit_image.size[0]),
                                  interpolation=T.InterpolationMode.BILINEAR,
                                  antialias=True)(new_edit_image)

        image = Image.composite(new_edit_image, edit_image, edit_mask)

        return self.trans_pil_tensor(image).unsqueeze(0).permute(0, 2, 3, 1)

    def preprocess(self,
                   reference_image=None,
                   edit_image=None,
                   edit_mask=None,
                   use_reference=True,
                   task_type=None,
                   height=1024,
                   width=1024,
                   keep_pixels_rate=0.8,
                   max_seq_length=4096):
        self.max_seq_len = max_seq_length
        if not use_reference and edit_image is not None:
            reference_image = None
        if edit_mask is not None and edit_image is not None:
            iH, iW = edit_image.shape[1:3]
            mH, mW = edit_mask.shape[1:3]
            if iH != mH or iW != mW:
                edit_mask = torch.ones(edit_image.shape[:3])

        if task_type != 'repainting':
            repainting_scale = 0
        else:
            repainting_scale = 1
        if task_type in self.task_list:
            edit_image = self.edit_preprocess(self.task_list[task_type]['ANNOTATOR'], 0,
                                              edit_image, edit_mask)
        if reference_image is not None:
            reference_image = self.image_check(reference_image) - 0.5
        if edit_image is not None:
            edit_image = self.image_check(edit_image) - 0.5
        # for reference generation
        if edit_image is None:
            edit_image = torch.zeros([3, height, width])
            edit_mask = torch.ones([1, height, width])
        else:
            if edit_mask is None:
                _, eH, eW = edit_image.shape
                edit_mask = np.ones((eH, eW))
            else:
                edit_mask = np.asarray(edit_mask)[0]
                edit_mask = np.where(edit_mask > 0.5, 1, 0)
            edit_mask = edit_mask.astype(
                np.float32) if np.any(edit_mask) else np.ones_like(edit_mask).astype(
                np.float32)
            edit_mask = torch.tensor(edit_mask).unsqueeze(0)

        edit_image = edit_image * (1 - edit_mask * repainting_scale)

        out_h, out_w = edit_image.shape[-2:]

        assert edit_mask is not None
        if reference_image is not None:
            _, H, W = reference_image.shape
            _, eH, eW = edit_image.shape
            if not True:
                # align height with edit_image
                scale = eH / H
                tH, tW = eH, int(W * scale)
                reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(
                    reference_image)
            else:
                # padding
                if H >= keep_pixels_rate * eH:
                    tH = int(eH * keep_pixels_rate)
                    scale = tH / H
                    tW = int(W * scale)
                    reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(
                        reference_image)
                rH, rW = reference_image.shape[-2:]
                delta_w = 0
                delta_h = eH - rH
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                reference_image = T.Pad(padding, fill=0, padding_mode="constant")(reference_image)
            edit_image = torch.cat([reference_image, edit_image], dim=-1)
            edit_mask = torch.cat([torch.zeros([1, reference_image.shape[1], reference_image.shape[2]]), edit_mask],
                                  dim=-1)
            slice_w = reference_image.shape[-1]
        else:
            slice_w = 0

        H, W = edit_image.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / self.d) * (W / self.d))))
        rH = int(H * scale) // self.d * self.d
        rW = int(W * scale) // self.d * self.d
        slice_w = int(slice_w * scale) // self.d * self.d

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)

        change_image = edit_image * edit_mask
        edit_image = edit_image * (1 - edit_mask)
        edit_image = edit_image.unsqueeze(0).permute(0, 2, 3, 1)
        change_image = change_image.unsqueeze(0).permute(0, 2, 3, 1)
        slice_w = slice_w if slice_w < 30 else slice_w + 30

        return edit_image + 0.5, change_image + 0.5, edit_mask, out_h, out_w, slice_w
