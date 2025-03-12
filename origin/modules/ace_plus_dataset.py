# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import re, io
import numpy as np
import random, torch
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.directory import osp_path
from scepter.modules.utils.file_system import FS
from torchvision.transforms import InterpolationMode
def load_image(prefix, img_path, cvt_type=None):
    if img_path is None or img_path == '':
        return None
    img_path = osp_path(prefix, img_path)
    with FS.get_object(img_path) as image_bytes:
        image = Image.open(io.BytesIO(image_bytes))
        if cvt_type is not None:
            image = pillow_convert(image, cvt_type)
    return image
def transform_image(image, std = 0.5, mean = 0.5):
    return (image.permute(2, 0, 1)/255. - mean)/std
def transform_mask(mask):
    return mask.unsqueeze(0)/255.
def ensure_src_align_target_h_mode(src_image, size, image_id, interpolation=InterpolationMode.BILINEAR):
    # padding mode
    H, W = size
    ret_image = []
    for one_id in image_id:
        edit_image = src_image[one_id]
        _, eH, eW = edit_image.shape
        scale = H/eH
        tH, tW = H, int(eW * scale)
        ret_image.append(T.Resize((tH, tW), interpolation=interpolation, antialias=True)(edit_image))
    return ret_image

def ensure_src_align_target_padding_mode(src_image, size, image_id, size_h = [], interpolation=InterpolationMode.BILINEAR):
    # padding mode
    H, W = size

    ret_data = []
    ret_h = []
    for idx, one_id in enumerate(image_id):
        if len(size_h) < 1:
            rH = random.randint(int(H / 3), int(H))
        else:
            rH = size_h[idx]
        ret_h.append(rH)
        edit_image = src_image[one_id]
        _, eH, eW = edit_image.shape
        scale = rH/eH
        tH, tW = rH, int(eW * scale)
        edit_image = T.Resize((tH, tW), interpolation=interpolation, antialias=True)(edit_image)
        # padding
        delta_w = 0
        delta_h = H - tH
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        ret_data.append(T.Pad(padding, fill=0, padding_mode="constant")(edit_image).float())
    return ret_data, ret_h

def ensure_limit_sequence(image, max_seq_len = 4096, d = 16, interpolation=InterpolationMode.BILINEAR):
    # resize image for max_seq_len, while keep the aspect ratio
    H, W = image.shape[-2:]
    scale = min(1.0, math.sqrt(max_seq_len / ((H / d) * (W / d))))
    rH = int(H * scale) // d * d  # ensure divisible by self.d
    rW = int(W * scale) // d * d
    # print(f"{H} {W} -> {rH} {rW}")
    image = T.Resize((rH, rW), interpolation=interpolation, antialias=True)(image)
    return image

@DATASETS.register_class()
class ACEPlusDataset(BaseDataset):
    para_dict = {
        "DELIMITER": {
            "value": "#;#",
            "description": "The delimiter for records of data list."
        },
        "FIELDS": {
            "value": ["data_type", "edit_image", "edit_mask", "ref_image", "target_image", "prompt"],
            "description": "The fields for every record."
        },
        "PATH_PREFIX": {
            "value": "",
            "description": "The path prefix for every input image."
        },
        "EDIT_TYPE_LIST": {
            "value": [],
            "description": "The edit type list to be trained for data list."
        },
        "MAX_SEQ_LEN": {
            "value": 4096,
            "description": "The max sequence length for input image."
        },
        "D": {
            "value": 16,
            "description": "Patch size for resized image."
        }
    }
    para_dict.update(BaseDataset.para_dict)
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        delimiter = cfg.get("DELIMITER", "#;#")
        fields = cfg.get("FIELDS", [])
        prefix = cfg.get("PATH_PREFIX", "")
        edit_type_list = cfg.get("EDIT_TYPE_LIST", [])
        self.modify_mode = cfg.get("MODIFY_MODE", True)
        self.max_seq_len = cfg.get("MAX_SEQ_LEN", 4096)
        self.repaiting_scale = cfg.get("REPAINTING_SCALE", 0.5)
        self.d = cfg.get("D", 16)
        prompt_file = cfg.DATA_LIST
        self.items = self.read_data_list(delimiter,
                                         fields,
                                         prefix,
                                         edit_type_list,
                                         prompt_file)
        random.shuffle(self.items)
        use_num = int(cfg.get('USE_NUM', -1))
        if use_num > 0:
            self.items = self.items[:use_num]
    def read_data_list(self, delimiter,
                             fields,
                             prefix,
                             edit_type_list,
                             prompt_file):
        with FS.get_object(prompt_file) as local_data:
            rows = local_data.decode('utf-8').strip().split('\n')
        items = list()
        dtype_level_num = {}
        for i, row in enumerate(rows):
            item = {"prefix": prefix}
            for key, val in zip(fields, row.split(delimiter)):
                item[key] = val
            edit_type = item["data_type"]
            if len(edit_type_list) > 0:
                for re_pattern in edit_type_list:
                    if re.match(re_pattern, edit_type):
                        items.append(item)
                        if edit_type not in dtype_level_num:
                            dtype_level_num[edit_type] = 0
                        dtype_level_num[edit_type] += 1
                        break
            else:
                items.append(item)
                if edit_type not in dtype_level_num:
                    dtype_level_num[edit_type] = 0
                dtype_level_num[edit_type] += 1
        for edit_type in dtype_level_num:
            self.logger.info(f"{edit_type} has {dtype_level_num[edit_type]} samples.")
        return items
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self._get(index)
        return self.pipeline(item)

    def _get(self, index):
        # normalize
        sample_id =  index%len(self)
        index = self.items[index%len(self)]
        prefix = index.get("prefix", "")
        edit_image = index.get("edit_image", "")
        edit_mask = index.get("edit_mask", "")
        ref_image = index.get("ref_image", "")
        target_image = index.get("target_image", "")
        prompt = index.get("prompt", "")

        edit_image = load_image(prefix, edit_image, cvt_type="RGB") if edit_image != "" else None
        edit_mask = load_image(prefix, edit_mask, cvt_type="L") if edit_mask != "" else None
        ref_image = load_image(prefix, ref_image, cvt_type="RGB") if ref_image != "" else None
        target_image = load_image(prefix, target_image, cvt_type="RGB") if target_image != "" else None
        assert target_image is not None

        edit_id, ref_id, src_image_list, src_mask_list = [], [], [], []
        # parse editing image
        if edit_image is None:
            edit_image = Image.new("RGB", target_image.size, (255, 255, 255))
            edit_mask = Image.new("L", edit_image.size, 255)
        elif edit_mask is None:
            edit_mask = Image.new("L", edit_image.size, 255)
        src_image_list.append(edit_image)
        edit_id.append(0)
        src_mask_list.append(edit_mask)
        # parse reference image
        if ref_image is not None:
            src_image_list.append(ref_image)
            ref_id.append(1)
            src_mask_list.append(Image.new("L", ref_image.size, 0))

        image = transform_image(torch.tensor(np.array(target_image).astype(np.float32)))
        if edit_mask is not None:
            image_mask = transform_mask(torch.tensor(np.array(edit_mask).astype(np.float32)))
        else:
            image_mask = Image.new("L", target_image.size, 255)
            image_mask = transform_mask(torch.tensor(np.array(image_mask).astype(np.float32)))


        src_image_list = [transform_image(torch.tensor(np.array(im).astype(np.float32))) for im in src_image_list]
        src_mask_list = [transform_mask(torch.tensor(np.array(im).astype(np.float32))) for im in src_mask_list]

        # decide the repainting scale for the editing task
        if len(ref_id) > 0:
            repainting_scale = 1.0
        else:
            repainting_scale = self.repaiting_scale
        for e_i in edit_id:
            src_image_list[e_i] = src_image_list[e_i] * (1 - repainting_scale * src_mask_list[e_i])
        size = image.shape[1:]
        ref_image_list, ret_h = ensure_src_align_target_padding_mode(src_image_list, size,
                                                                                   image_id=ref_id,
                                                                                   interpolation=InterpolationMode.NEAREST_EXACT)
        ref_mask_list, ret_h = ensure_src_align_target_padding_mode(src_mask_list, size,
                                                                                  size_h=ret_h,
                                                                                  image_id=ref_id,
                                                                                  interpolation=InterpolationMode.NEAREST_EXACT)

        edit_image_list = ensure_src_align_target_h_mode(src_image_list, size,
                                                                       image_id=edit_id,
                                                                       interpolation=InterpolationMode.NEAREST_EXACT)
        edit_mask_list = ensure_src_align_target_h_mode(src_mask_list, size,
                                                                      image_id=edit_id,
                                                                      interpolation=InterpolationMode.NEAREST_EXACT)



        src_image_list = [torch.cat(ref_image_list + edit_image_list, dim=-1)]
        src_mask_list = [torch.cat(ref_mask_list + edit_mask_list, dim=-1)]
        image = torch.cat(ref_image_list + [image], dim=-1)
        image_mask = torch.cat(ref_mask_list + [image_mask], dim=-1)

        # limit max sequence length
        image = ensure_limit_sequence(image, max_seq_len = self.max_seq_len,
                                      d = self.d, interpolation=InterpolationMode.NEAREST_EXACT)
        image_mask = ensure_limit_sequence(image_mask, max_seq_len = self.max_seq_len,
                                      d = self.d, interpolation=InterpolationMode.NEAREST_EXACT)
        src_image_list = [ensure_limit_sequence(i, max_seq_len = self.max_seq_len,
                                      d = self.d, interpolation=InterpolationMode.NEAREST_EXACT) for i in src_image_list]
        src_mask_list = [ensure_limit_sequence(i, max_seq_len = self.max_seq_len,
                                      d = self.d, interpolation=InterpolationMode.NEAREST_EXACT) for i in src_mask_list]

        if self.modify_mode:
            # To be modified regions according to mask
            modify_image_list = [ii * im for ii, im in zip(src_image_list, src_mask_list)]
            # To be edited regions according to mask
            src_image_list = [ii * (1 - im) for ii, im in zip(src_image_list, src_mask_list)]
        else:
            src_image_list = src_image_list
            modify_image_list = src_image_list

        item = {
            "src_image_list": src_image_list,
            "src_mask_list": src_mask_list,
            "modify_image_list": modify_image_list,
            "image": image,
            "image_mask": image_mask,
            "edit_id": edit_id,
            "ref_id": ref_id,
            "prompt": prompt,
            "edit_key": index["edit_key"] if "edit_key" in index else "",
            "sample_id": sample_id
        }
        return item

    @staticmethod
    def collate_fn(batch):
        collect = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                collect[k].append(v)
        new_batch = dict()
        for k, v in collect.items():
            if all([i is None for i in v]):
                new_batch[k] = None
            else:
                new_batch[k] = v
        return new_batch
