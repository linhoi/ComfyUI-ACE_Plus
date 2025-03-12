# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from collections import OrderedDict

import torch, numpy as np
from PIL import Image
from scepter.modules.model.registry import MODELS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from .registry import BaseInference, INFERENCES
from .utils import ACEPlusImageProcessor

@INFERENCES.register_class()
class ACEInference(BaseInference):
    '''
        reuse the ldm code
    '''
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self.pipe = MODELS.build(cfg.MODEL, logger=self.logger).eval().to(we.device_id)
        self.image_processor = ACEPlusImageProcessor(max_seq_len=cfg.MAX_SEQ_LEN)
        self.input = {k.lower(): dict(v).get('DEFAULT', None) if isinstance(v, (dict, OrderedDict, Config)) else v for
                      k, v in cfg.SAMPLE_ARGS.items()}
        self.dtype = getattr(torch, cfg.get("DTYPE", "bfloat16"))
    @torch.no_grad()
    def __call__(self,
                 reference_image=None,
                 edit_image=None,
                 edit_mask=None,
                 prompt='',
                 edit_type=None,
                 output_height=1024,
                 output_width=1024,
                 sampler='flow_euler',
                 sample_steps=28,
                 guide_scale=50,
                 lora_path=None,
                 seed=-1,
                 repainting_scale=0,
                 use_change=False,
                 keep_pixels=False,
                 keep_pixels_rate=0.8,
                 **kwargs):
        # convert the input info to the input of ldm.
        if isinstance(prompt, str):
            prompt = [prompt]
        seed = seed if seed >= 0 else random.randint(0, 2 ** 24 - 1)
        image, mask, change_image, content_image, out_h, out_w, slice_w = self.image_processor.preprocess(reference_image, edit_image, edit_mask,
                                                                             height=output_height, width=output_width,
                                                                             repainting_scale=repainting_scale,
                                                                             keep_pixels=keep_pixels,
                                                                             keep_pixels_rate=keep_pixels_rate,
                                                                             use_change = use_change)
        change_image = [None] if change_image is None else [change_image.to(we.device_id)]
        image, mask = [image.to(we.device_id)], [mask.to(we.device_id)]

        (src_image_list, src_mask_list, modify_image_list,
         edit_id, prompt) = [image], [mask], [change_image], [[0]], [prompt]

        with torch.amp.autocast(enabled=True, dtype=self.dtype, device_type='cuda'):
            out_image = self.pipe(
                src_image_list=src_image_list,
                modify_image_list= modify_image_list,
                src_mask_list=src_mask_list,
                edit_id=edit_id,
                image=image,
                image_mask=mask,
                prompt=prompt,
                sampler='flow_euler',
                sample_steps=sample_steps,
                seed=seed,
                guide_scale=guide_scale,
                show_process=True,
            )
        imgs = [x_i['reconstruct_image'].float().permute(1, 2, 0).cpu().numpy()
            for x_i in out_image
        ]
        imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
        edit_image = Image.fromarray((torch.clamp(image[0] / 2 + 0.5, min=0.0, max=1.0)*255).float().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        change_image = Image.fromarray((torch.clamp(change_image[0] / 2 + 0.5, min=0.0, max=1.0)*255).float().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        mask = Image.fromarray((mask[0] * 255).squeeze(0).cpu().numpy().astype(np.uint8))
        return self.image_processor.postprocess(imgs[0], slice_w, out_w, out_h), edit_image, change_image, mask, seed
