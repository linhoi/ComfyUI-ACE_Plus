# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from collections import OrderedDict

import torch, os
from diffusers import FluxFillPipeline
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from transformers import T5TokenizerFast
from .utils import ACEPlusImageProcessor

class ACEPlusDiffuserInference():
    def __init__(self, logger=None):
        if logger is None:
            logger = get_logger(name='ace_plus')
        self.logger = logger
        self.input = {}

    def load_default(self, cfg):
        if cfg is not None:
            self.input_cfg = {k.lower(): v for k, v in cfg.INPUT.items()}
            self.input = {k.lower(): dict(v).get('DEFAULT', None) if isinstance(v, (dict, OrderedDict, Config)) else v for k, v in cfg.INPUT.items()}
            self.output = {k.lower(): v for k, v in cfg.OUTPUT.items()}

    def init_from_cfg(self, cfg):
        self.max_seq_len = cfg.get("MAX_SEQ_LEN", 4096)
        self.image_processor = ACEPlusImageProcessor(max_seq_len=self.max_seq_len)

        local_folder = FS.get_dir_to_local_dir(cfg.MODEL.PRETRAINED_MODEL)

        self.pipe = FluxFillPipeline.from_pretrained(local_folder, torch_dtype=torch.bfloat16).to(we.device_id)

        tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(local_folder, "tokenizer_2"),
                                                      additional_special_tokens=["{image}"])
        self.pipe.tokenizer_2 = tokenizer_2
        self.load_default(cfg.DEFAULT_PARAS)

    def prepare_input(self,
                      image,
                      mask,
                      batch_size=1,
                      dtype = torch.bfloat16,
                      num_images_per_prompt=1,
                      height=512,
                      width=512,
                      generator=None):
        num_channels_latents = self.pipe.vae.config.latent_channels
        # import pdb;pdb.set_trace()
        mask, masked_image_latents = self.pipe.prepare_mask_latents(
            mask.unsqueeze(0),
            image.unsqueeze(0).to(we.device_id, dtype = dtype),
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            dtype,
            we.device_id,
            generator,
        )
        # import pdb;pdb.set_trace()
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
        return masked_image_latents

    @torch.no_grad()
    def __call__(self,
                 reference_image=None,
                 edit_image=None,
                 edit_mask=None,
                 prompt='',
                 task=None,
                 output_height=1024,
                 output_width=1024,
                 sampler='flow_euler',
                 sample_steps=28,
                 guide_scale=50,
                 lora_path=None,
                 seed=-1,
                 tar_index=0,
                 align=0,
                 repainting_scale=0,
                 **kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]
        seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        # edit_image, edit_mask, change_image, content_image, out_h, out_w, slice_w
        image, mask, _, _, out_h, out_w, slice_w = self.image_processor.preprocess(reference_image, edit_image, edit_mask,
                                                                             width = output_width,
                                                                             height = output_height,
                                                                             repainting_scale = repainting_scale)
        h, w = image.shape[1:]
        generator = torch.Generator("cpu").manual_seed(seed)
        masked_image_latents = self.prepare_input(image, mask,
                                               batch_size=len(prompt) , height=h, width=w, generator = generator)

        if lora_path is not None:
            with FS.get_from(lora_path) as local_path:
                self.pipe.load_lora_weights(local_path)



        image = self.pipe(
            prompt=prompt,
            masked_image_latents=masked_image_latents,
            height=h,
            width=w,
            guidance_scale=guide_scale,
            num_inference_steps=sample_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        if lora_path is not None:
            self.pipe.unload_lora_weights()
        return self.image_processor.postprocess(image, slice_w, out_w, out_h), seed


if __name__ == '__main__':
    pass