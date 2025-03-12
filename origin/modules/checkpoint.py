# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os, torch
import os.path as osp
import warnings
from collections import OrderedDict
from safetensors.torch import save_file
from scepter.modules.solver.hooks import CheckpointHook, BackwardHook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

_DEFAULT_CHECKPOINT_PRIORITY = 300

def convert_to_comfyui_lora(ori_sd, prefix = "lora_unet"):
    new_ckpt = OrderedDict()
    for k,v in ori_sd.items():
        new_k = k.replace(".lora_A.0_SwiftLoRA.", ".lora_down.").replace(".lora_B.0_SwiftLoRA.", ".lora_up.")
        new_k = prefix + "_" + new_k.split(".lora")[0].replace("model.", "").replace(".", "_") + ".lora" + new_k.split(".lora")[1]
        alpha_k = new_k.split(".lora")[0] + ".alpha"
        new_ckpt[new_k] = v
        if "lora_up" in new_k:
            alpha = v.shape[-1]
        elif "lora_down" in new_k:
            alpha = v.shape[0]
        new_ckpt[alpha_k] = torch.tensor(float(alpha)).to(v)
    return new_ckpt

@HOOKS.register_class()
class ACECheckpointHook(CheckpointHook):
    """ Checkpoint resume or save hook.
    Args:
        interval (int): Save interval, by epoch.
        save_best (bool): Save the best checkpoint by a metric key, default is False.
        save_best_by (str): How to get the best the checkpoint by the metric key, default is ''.
            + means the higher the best (default).
            - means the lower the best.
            E.g. +acc@1, -err@1, acc@5(same as +acc@5)
    """

    def __init__(self, cfg, logger=None):
        super(ACECheckpointHook, self).__init__(cfg, logger=logger)

    def after_iter(self, solver):
        super().after_iter(solver)
        if solver.total_iter != 0 and (
            (solver.total_iter + 1) % self.interval == 0
                or solver.total_iter == solver.max_steps - 1):
            from swift import SwiftModel
            if isinstance(solver.model, SwiftModel) or (
                    hasattr(solver.model, 'module')
                    and isinstance(solver.model.module, SwiftModel)):
                save_path = osp.join(
                    solver.work_dir,
                    'checkpoints/{}-{}'.format(self.save_name_prefix,
                                               solver.total_iter + 1))
                if we.rank == 0:
                    tuner_model = os.path.join(save_path, '0_SwiftLoRA', 'adapter_model.bin')
                    save_model = os.path.join(save_path, '0_SwiftLoRA', 'comfyui_model.safetensors')
                    if FS.exists(tuner_model):
                        with FS.get_from(tuner_model) as local_file:
                            swift_lora_sd = torch.load(local_file, weights_only=True)
                        safetensor_lora_sd = convert_to_comfyui_lora(swift_lora_sd)
                        with FS.put_to(save_model) as local_file:
                            save_file(safetensor_lora_sd, local_file)
    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            ACECheckpointHook.para_dict,
                            set_name=True)

@HOOKS.register_class()
class ACEBackwardHook(BackwardHook):
    def grad_clip(self, optimizer):
        for params_group in optimizer.param_groups:
            train_params = []
            for param in params_group['params']:
                if param.requires_grad:
                    train_params.append(param)
            # print(len(train_params), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(parameters=train_params,
                                       max_norm=self.gradient_clip)

    def after_iter(self, solver):
        if solver.optimizer is not None and solver.is_train_mode:
            if solver.loss is None:
                warnings.warn(
                    'solver.loss should not be None in train mode, remember to call solver._reduce_scalar()!'
                )
                return
            if solver.scaler is not None:
                solver.scaler.scale(solver.loss /
                                    self.accumulate_step).backward()
                self.current_step += 1
                # Suppose profiler run after backward, so we need to set backward_prev_step
                # as the previous one step before the backward step
                if self.current_step % self.accumulate_step == 0:
                    solver.scaler.unscale_(solver.optimizer)
                    if self.gradient_clip > 0:
                        self.grad_clip(solver.optimizer)
                    self.profile(solver)
                    solver.scaler.step(solver.optimizer)
                    solver.scaler.update()
                    solver.optimizer.zero_grad()
            else:
                (solver.loss / self.accumulate_step).backward()
                self.current_step += 1
                # Suppose profiler run after backward, so we need to set backward_prev_step
                # as the previous one step before the backward step
                if self.current_step % self.accumulate_step == 0:
                    if self.gradient_clip > 0:
                        self.grad_clip(solver.optimizer)
                    self.profile(solver)
                    solver.optimizer.step()
                    solver.optimizer.zero_grad()
            if solver.lr_scheduler:
                if self.current_step % self.accumulate_step == 0:
                    solver.lr_scheduler.step()
            if self.current_step % self.accumulate_step == 0:
                setattr(solver, 'backward_step', True)
                self.current_step = 0
            else:
                setattr(solver, 'backward_step', False)
            solver.loss = None
        if self.empty_cache_step > 0 and solver.total_iter % self.empty_cache_step == 0:
            torch.cuda.empty_cache()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            ACEBackwardHook.para_dict,
                            set_name=True)
