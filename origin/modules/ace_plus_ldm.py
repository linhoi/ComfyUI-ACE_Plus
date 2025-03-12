# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
import copy
import math
import random
from contextlib import nullcontext
from einops import rearrange
from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.registry import MODELS, DIFFUSIONS, BACKBONES, LOSSES, TOKENIZERS, EMBEDDERS
from scepter.modules.model.utils.basic_utils import check_list_of_list, to_device, pack_imagelist_into_tensor, \
    limit_batch_data, unpack_tensor_into_imagelist, count_params, disabled_train
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we

@MODELS.register_class()
class LatentDiffusionACEPlus(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.guide_scale = cfg.get('GUIDE_SCALE', 1.0)

    def init_params(self):
        self.parameterization = self.cfg.get('PARAMETERIZATION', 'rf')
        assert self.parameterization in [
            'eps', 'x0', 'v', 'rf'
        ], 'currently only supporting "eps" and "x0" and "v" and "rf"'

        diffusion_cfg = self.cfg.get("DIFFUSION", None)
        assert diffusion_cfg is not None
        if self.cfg.have("WORK_DIR"):
            diffusion_cfg.WORK_DIR = self.cfg.WORK_DIR
        self.diffusion = DIFFUSIONS.build(diffusion_cfg, logger=self.logger)

        self.pretrained_model = self.cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = self.cfg.get('IGNORE_KEYS', [])

        self.model_config = self.cfg.DIFFUSION_MODEL
        self.first_stage_config = self.cfg.FIRST_STAGE_MODEL
        self.cond_stage_config = self.cfg.COND_STAGE_MODEL
        self.tokenizer_config = self.cfg.get('TOKENIZER', None)
        self.loss_config = self.cfg.get('LOSS', None)

        self.scale_factor = self.cfg.get('SCALE_FACTOR', 0.18215)
        self.size_factor = self.cfg.get('SIZE_FACTOR', 16)
        self.default_n_prompt = self.cfg.get('DEFAULT_N_PROMPT', '')
        self.default_n_prompt = '' if self.default_n_prompt is None else self.default_n_prompt
        self.p_zero = self.cfg.get('P_ZERO', 0.0)
        self.train_n_prompt = self.cfg.get('TRAIN_N_PROMPT', '')
        if self.default_n_prompt is None:
            self.default_n_prompt = ''
        if self.train_n_prompt is None:
            self.train_n_prompt = ''
        self.use_ema = self.cfg.get('USE_EMA', False)
        self.model_ema_config = self.cfg.get('DIFFUSION_MODEL_EMA', None)

    def construct_network(self):
        # embedding_context = torch.device("meta") if self.model_config.get("PRETRAINED_MODEL", None) else nullcontext()
        # with embedding_context:
        self.model = BACKBONES.build(self.model_config, logger=self.logger).to(torch.bfloat16)
        self.logger.info('all parameters:{}'.format(count_params(self.model)))
        if self.use_ema:
            if self.model_ema_config:
                self.model_ema = BACKBONES.build(self.model_ema_config,
                                                 logger=self.logger)
            else:
                self.model_ema = copy.deepcopy(self.model)
            self.model_ema = self.model_ema.eval()
            for param in self.model_ema.parameters():
                param.requires_grad = False
        if self.loss_config:
            self.loss = LOSSES.build(self.loss_config, logger=self.logger)
        if self.tokenizer_config is not None:
            self.tokenizer = TOKENIZERS.build(self.tokenizer_config,
                                              logger=self.logger)
        if self.first_stage_config:
            self.first_stage_model = MODELS.build(self.first_stage_config,
                                                  logger=self.logger)
            self.first_stage_model = self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        else:
            self.first_stage_model = None
        if self.tokenizer_config is not None:
            self.cond_stage_config.KWARGS = {
                'vocab_size': self.tokenizer.vocab_size
            }
        if self.cond_stage_config == '__is_unconditional__':
            print(
                f'Training {self.__class__.__name__} as an unconditional model.'
            )
            self.cond_stage_model = None
        else:
            model = EMBEDDERS.build(self.cond_stage_config, logger=self.logger)
            self.cond_stage_model = model.eval().requires_grad_(False)
            self.cond_stage_model.train = disabled_train

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        def run_one_image(u):
            zu = self.first_stage_model.encode(u)
            if isinstance(zu, (tuple, list)):
                zu = zu[0]
            return zu

        z = [run_one_image(u.unsqueeze(0) if u.dim() == 3 else u) for u in x]
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        return [self.first_stage_model.decode(zu) for zu in z]
    def noise_sample(self, num_samples, h, w, seed, dtype=torch.bfloat16):
        noise = torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(h / 16),
            2 * math.ceil(w / 16),
            device=we.device_id,
            dtype=dtype,
            generator=torch.Generator(device=we.device_id).manual_seed(seed),
        )
        return noise
    def resize_func(self, x, size):
        if x is None: return x
        return F.interpolate(x.unsqueeze(0), size = size, mode='nearest-exact')
    def parse_ref_and_edit(self, src_image,
                           modify_image,
                           src_image_mask,
                           text_embedding,
                           #text_mask,
                           edit_id):
        edit_image = []
        modi_image = []
        edit_mask = []
        ref_image = []
        ref_mask = []
        ref_context = []
        ref_y = []
        ref_id = []
        txt = []
        txt_y = []
        for sample_id, (one_src,
                        one_modify,
                        one_src_mask,
                        one_text_embedding,
                        one_text_y,
                        # one_text_mask,
                        one_edit_id)  in enumerate(zip(src_image,
                                        modify_image,
                                        src_image_mask,
                                        text_embedding["context"],
                                        text_embedding["y"],
                                        #text_mask,
                                        edit_id)
                                ):
            ref_id.append([i for i in range(len(one_src))])
            if hasattr(self, "ref_cond_stage_model") and self.ref_cond_stage_model:
                ref_image.append(self.ref_cond_stage_model.encode_list([((i + 1.0) / 2.0 * 255).type(torch.uint8) for i in one_src]))
            else:
                ref_image.append(one_src)
            ref_mask.append(one_src_mask)
            # process edit image & edit image mask
            current_edit_image = to_device([one_src[i] for i in one_edit_id], strict=False)
            current_edit_image = [v.squeeze(0) for v in self.encode_first_stage(current_edit_image)]
            # process modi image
            current_modify_image = to_device([one_modify[i] for i in one_edit_id],
                                             strict=False)
            current_modify_image = [
                v.squeeze(0)
                for v in self.encode_first_stage(current_modify_image)
            ]
            current_edit_image_mask = to_device(
                [one_src_mask[i] for i in one_edit_id], strict=False)
            current_edit_image_mask = [
                self.reshape_func(m).squeeze(0)
                for m in current_edit_image_mask
            ]

            edit_image.append(current_edit_image)
            modi_image.append(current_modify_image)
            edit_mask.append(current_edit_image_mask)
            ref_context.append(one_text_embedding[:len(ref_id[-1])])
            ref_y.append(one_text_y[:len(ref_id[-1])])
        if not sum(len(src_) for src_ in src_image) > 0:
            ref_image = None
            ref_context = None
            ref_y = None
        for sample_id, (one_text_embedding, one_text_y) in enumerate(zip(text_embedding["context"],
                                                    text_embedding["y"])):
            txt.append(one_text_embedding[-1].squeeze(0))
            txt_y.append(one_text_y[-1])
        return {
            "edit": edit_image,
            'modify': modi_image,
            "edit_mask": edit_mask,
            "edit_id": edit_id,
            "ref_context": ref_context,
            "ref_y": ref_y,
            "context": txt,
            "y": txt_y,
            "ref_x": ref_image,
            "ref_mask": ref_mask,
            "ref_id": ref_id
        }


    def reshape_func(self, mask):
        mask = mask.to(torch.bfloat16)
        mask = mask.view((-1, mask.shape[-2], mask.shape[-1]))
        mask = rearrange(
            mask,
            "c (h ph) (w pw) -> c (ph pw) h w",
            ph=8,
            pw=8,
        )
        return mask

    def forward_train(self,
                      src_image_list=[],
                      modify_image_list=[],
                      src_mask_list=[],
                      edit_id=[],
                      image=None,
                      image_mask=None,
                      noise=None,
                      prompt=[],
                      **kwargs):
        '''
           Args:
               src_image: list of list of src_image
               src_image_mask: list of list of src_image_mask
               image: target image
               image_mask: target image mask
               noise: default is None, generate automaticly
               ref_prompt: list of list of text
               prompt: list of text
               **kwargs:
           Returns:
        '''
        assert check_list_of_list(src_image_list) and check_list_of_list(
            src_mask_list)
        assert self.cond_stage_model is not None

        gc_seg = kwargs.pop("gc_seg", [])
        gc_seg = int(gc_seg[0]) if len(gc_seg) > 0 else 0
        align = kwargs.pop("align", [])
        prompt_ = [[pp] if isinstance(pp, str) else pp for pp in prompt]
        if len(align) < 1: align = [0] * len(prompt_)
        context = getattr(self.cond_stage_model, 'encode_list_of_list')(prompt_)
        guide_scale = self.guide_scale
        if guide_scale is not None:
            guide_scale = torch.full((len(prompt_),), guide_scale, device=we.device_id)
        else:
            guide_scale = None
        # image and image_mask
        # print("is list of list", check_list_of_list(image))
        if check_list_of_list(image):
            image = [to_device(ix) for ix in image]
            x_start = [self.encode_first_stage(ix, **kwargs) for ix in image]
            noise = [[torch.randn_like(ii) for ii in ix] for ix in x_start]
            x_start = [torch.cat(ix, dim=-1) for ix in x_start]
            noise = [torch.cat(ix, dim=-1) for ix in noise]

            noise, _ = pack_imagelist_into_tensor(noise)

            image_mask = [to_device(im, strict=False) for im in image_mask]
            x_mask = [[self.reshape_func(i).squeeze(0) for i in im] if im is not None else [None] * len(ix) for ix, im in zip(image, image_mask)]
            x_mask = [torch.cat(im, dim=-1) for im in x_mask]
        else:
            image = to_device(image)
            x_start = self.encode_first_stage(image, **kwargs)
            image_mask = to_device(image_mask, strict=False)
            x_mask = [self.reshape_func(i).squeeze(0) for i in image_mask] if image_mask is not None else [None] * len(
                image)
        loss_mask, _ = pack_imagelist_into_tensor(
            tuple(torch.ones_like(ix, dtype=torch.bool, device=ix.device) for ix in x_start))
        x_start, x_shapes = pack_imagelist_into_tensor(x_start)
        context['x_shapes'] = x_shapes
        context['align'] = align
        # process image mask

        context['x_mask'] = x_mask
        ref_edit_context = self.parse_ref_and_edit(src_image_list, modify_image_list, src_mask_list, context, edit_id)
        context.update(ref_edit_context)

        teacher_context = copy.deepcopy(context)
        teacher_context["context"] = torch.cat(teacher_context["context"], dim=0)
        teacher_context["y"] = torch.cat(teacher_context["y"], dim=0)
        loss = self.diffusion.loss(x_0=x_start,
                                   model=self.model,
                                   model_kwargs={"cond": context,
                                                 "gc_seg": gc_seg,
                                                 "guidance": guide_scale},
                                   noise=noise,
                                   reduction='none',
                                   **kwargs)
        loss = loss[loss_mask].mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret

    @torch.no_grad()
    def forward_test(self,
                     src_image_list=[],
                     modify_image_list=[],
                     src_mask_list=[],
                     edit_id=[],
                     image=None,
                     image_mask=None,
                     prompt=[],
                     sampler='flow_euler',
                     sample_steps=20,
                     seed=2023,
                     guide_scale=3.5,
                     guide_rescale=0.0,
                     show_process=False,
                     log_num=-1,
                     **kwargs):
        outputs = self.forward_editing(
            src_image_list=src_image_list,
            src_mask_list=src_mask_list,
            modify_image_list=modify_image_list,
            edit_id=edit_id,
            image=image,
            image_mask=image_mask,
            prompt=prompt,
            sampler=sampler,
            sample_steps=sample_steps,
            seed=seed,
            guide_scale=guide_scale,
            guide_rescale=guide_rescale,
            show_process=show_process,
            log_num=log_num,
            **kwargs
        )
        return outputs

    @torch.no_grad()
    def forward_editing(self,
                        src_image_list=[],
                        modify_image_list=None,
                        src_mask_list=[],
                        edit_id=[],
                        image=None,
                        image_mask=None,
                        prompt=[],
                        sampler='flow_euler',
                        sample_steps=20,
                        seed=2023,
                        guide_scale=3.5,
                        log_num=-1,
                        **kwargs
                        ):
        # gc_seg is unused
        prompt, image, image_mask, src_image, modify_image,  src_image_mask, edit_id = limit_batch_data(
            [prompt, image, image_mask, src_image_list, modify_image_list, src_mask_list, edit_id], log_num)
        assert check_list_of_list(src_image) and check_list_of_list(src_image_mask)
        assert self.cond_stage_model is not None
        align = kwargs.pop("align", [])
        prompt_ = [[pp] if isinstance(pp, str) else pp for pp in prompt]
        if len(align) < 1: align = [0] * len(prompt_)
        context = getattr(self.cond_stage_model, 'encode_list_of_list')(prompt_)
        guide_scale = guide_scale or self.guide_scale
        if guide_scale is not None:
            guide_scale = torch.full((len(prompt),), guide_scale, device=we.device_id)
        else:
            guide_scale = None
        # image and image_mask
        seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        if image is not None:
            if check_list_of_list(image):
                image = [torch.cat(ix, dim=-1) for ix in image]
                image_mask = [torch.cat(im, dim=-1) for im in image_mask]
            noise = [self.noise_sample(1, ix.shape[1], ix.shape[2], seed) for ix in image]
        else:
            height, width = kwargs.pop("height"), kwargs.pop("width")
            noise = [self.noise_sample(1, height, width, seed) for _ in prompt]
        noise, x_shapes = pack_imagelist_into_tensor(noise)
        context['x_shapes'] = x_shapes
        context['align'] = align
        # process image mask
        image_mask = to_device(image_mask, strict=False)
        x_mask = [self.reshape_func(i).squeeze(0) for i in image_mask]
        context['x_mask'] = x_mask
        ref_edit_context = self.parse_ref_and_edit(src_image, modify_image, src_image_mask, context, edit_id)
        context.update(ref_edit_context)
        # UNet use input n_prompt
        # model = self.model_ema if self.use_ema and self.eval_ema else self.model
        # import pdb;pdb.set_trace()
        model = self.model
        embedding_context = model.no_sync if isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel) \
            else nullcontext
        with embedding_context():
            samples = self.diffusion.sample(
                noise=noise,
                sampler=sampler,
                model=self.model,
                model_kwargs={"cond": context, "guidance": guide_scale, "gc_seg": -1
                              },
                steps=sample_steps,
                show_progress=True,
                guide_scale=guide_scale,
                return_intermediate=None,
                **kwargs).float()
        samples = unpack_tensor_into_imagelist(samples, x_shapes)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_samples = self.decode_first_stage(samples)
        outputs = list()
        for i in range(len(prompt)):
            rec_img = torch.clamp((x_samples[i].float() + 1.0) / 2.0, min=0.0, max=1.0)
            rec_img = rec_img.squeeze(0)
            edit_imgs, modify_imgs, edit_img_masks = [], [], []
            if src_image is not None and src_image[i] is not None:
                if src_image_mask[i] is None:
                    src_image_mask[i] = [None] * len(src_image[i])
                for edit_img, modify_img, edit_mask in zip(src_image[i],  modify_image_list[i], src_image_mask[i]):
                    edit_img = torch.clamp((edit_img.float() + 1.0) / 2.0, min=0.0, max=1.0)
                    edit_imgs.append(edit_img.squeeze(0))
                    modify_img = torch.clamp((modify_img.float() + 1.0) / 2.0,
                                           min=0.0,
                                           max=1.0)
                    modify_imgs.append(modify_img.squeeze(0))
                    if edit_mask is None:
                        edit_mask = torch.ones_like(edit_img[[0], :, :])
                    edit_img_masks.append(edit_mask)
            one_tup = {
                'reconstruct_image': rec_img,
                'instruction': prompt[i],
                'edit_image': edit_imgs if len(edit_imgs) > 0 else None,
                'modify_image': modify_imgs if len(modify_imgs) > 0 else None,
                'edit_mask': edit_img_masks if len(edit_imgs) > 0 else None
            }
            if image is not None:
                if image_mask is None:
                    image_mask = [None] * len(image)
                ori_img = torch.clamp((image[i] + 1.0) / 2.0, min=0.0, max=1.0)
                one_tup['target_image'] = ori_img.squeeze(0)
                one_tup['target_mask'] = image_mask[i] if image_mask[i] is not None else torch.ones_like(
                    ori_img[[0], :, :])
            outputs.append(one_tup)
        return outputs

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionACEPlus.para_dict,
                            set_name=True)

