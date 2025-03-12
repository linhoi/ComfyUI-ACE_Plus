# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torchvision.transforms as T
import numpy as np
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from PIL import Image


def edit_preprocess(processor, device, edit_image, edit_mask):
    if edit_image is None or processor is None:
        return edit_image
    processor = Config(cfg_dict=processor, load=False)
    processor = ANNOTATORS.build(processor).to(device)
    new_edit_image = processor(np.asarray(edit_image))
    processor = processor.to("cpu")
    del processor
    new_edit_image = Image.fromarray(new_edit_image)
    return Image.composite(new_edit_image, edit_image, edit_mask)

class ACEPlusImageProcessor():
    def __init__(self, max_aspect_ratio=4, d=16, max_seq_len=1024):
        self.max_aspect_ratio = max_aspect_ratio
        self.d = d
        self.max_seq_len = max_seq_len
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def image_check(self, image):
        if image is None:
            return image
        # preprocess
        W, H = image.size
        if H / W > self.max_aspect_ratio:
            image = T.CenterCrop([int(self.max_aspect_ratio * W), W])(image)
        elif W / H > self.max_aspect_ratio:
            image = T.CenterCrop([H, int(self.max_aspect_ratio * H)])(image)
        return self.transforms(image)


    def preprocess(self,
                   reference_image=None,
                   edit_image=None,
                   edit_mask=None,
                   height=1024,
                   width=1024,
                   repainting_scale = 1.0,
                   keep_pixels = False,
                   keep_pixels_rate = 0.8,
                   use_change = False):
        reference_image = self.image_check(reference_image)
        edit_image = self.image_check(edit_image)
        # for reference generation
        if edit_image is None:
            edit_image = torch.zeros([3, height, width])
            edit_mask = torch.ones([1, height, width])
        else:
            if edit_mask is None:
                _, eH, eW = edit_image.shape
                edit_mask = np.ones((eH, eW))
            else:
                edit_mask = np.asarray(edit_mask)
                edit_mask = np.where(edit_mask > 128, 1, 0)
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
            if not keep_pixels:
                # align height with edit_image
                scale = eH / H
                tH, tW = eH, int(W * scale)
                reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(
                    reference_image)
            else:
                # padding
                if H >= keep_pixels_rate * eH:
                    tH = int(eH * keep_pixels_rate)
                    scale = tH/H
                    tW = int(W * scale)
                    reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(
                        reference_image)
                rH, rW = reference_image.shape[-2:]
                delta_w = 0
                delta_h = eH - rH
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                reference_image = T.Pad(padding, fill=0, padding_mode="constant")(reference_image)
            edit_image = torch.cat([reference_image, edit_image], dim=-1)
            edit_mask = torch.cat([torch.zeros([1, reference_image.shape[1], reference_image.shape[2]]), edit_mask], dim=-1)
            slice_w = reference_image.shape[-1]
        else:
            slice_w = 0

        H, W = edit_image.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / self.d) * (W / self.d))))
        rH = int(H * scale) // self.d * self.d  # ensure divisible by self.d
        rW = int(W * scale) // self.d * self.d
        slice_w = int(slice_w * scale) // self.d * self.d

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)
        content_image = edit_image
        if use_change:
            change_image = edit_image * edit_mask
            edit_image = edit_image * (1 - edit_mask)
        else:
            change_image = None
        return edit_image, edit_mask, change_image, content_image, out_h, out_w, slice_w


    def postprocess(self, image, slice_w, out_w, out_h):
        w, h = image.size
        if slice_w > 0:
            output_image = image.crop((slice_w + 30, 0, w, h))
            output_image = output_image.resize((out_w, out_h))
        else:
            output_image = image
        return output_image