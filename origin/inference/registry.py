# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from PIL.Image import Image
from collections import OrderedDict
from scepter.modules.utils.distribute import we
from scepter.modules.utils.config import Config
from scepter.modules.utils.logger import get_logger
from scepter.studio.utils.env import get_available_memory
from scepter.modules.model.registry import MODELS, BACKBONES, EMBEDDERS
from scepter.modules.utils.registry import Registry, build_from_config
def get_model(model_tuple):
    assert 'model' in model_tuple
    return model_tuple['model']

class BaseInference():
    '''
        support to load the components dynamicly.
        create and load model when run this model at the first time.
    '''
    def __init__(self, cfg, logger=None):
        if logger is None:
            logger = get_logger(name='scepter')
        self.logger = logger
        self.name = cfg.NAME

    def init_from_modules(self, modules):
        for k, v in modules.items():
            self.__setattr__(k, v)

    def infer_model(self, cfg, module_paras=None):
        module = {
            'model': None,
            'cfg': cfg,
            'device': 'offline',
            'name': cfg.NAME,
            'function_info': {},
            'paras': {}
        }
        if module_paras is None:
            return module
        function_info = {}
        paras = {
            k.lower(): v
            for k, v in module_paras.get('PARAS', {}).items()
        }
        for function in module_paras.get('FUNCTION', []):
            input_dict = {}
            for inp in function.get('INPUT', []):
                if inp.lower() in self.input:
                    input_dict[inp.lower()] = self.input[inp.lower()]
            function_info[function.NAME] = {
                'dtype': function.get('DTYPE', 'float32'),
                'input': input_dict
            }
        module['paras'] = paras
        module['function_info'] = function_info
        return module

    def init_from_ckpt(self, path, model, ignore_keys=list()):
        if path.endswith('safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(path)
        else:
            sd = torch.load(path, map_location='cpu', weights_only=True)

        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info(
                            'Ignore key {} from state_dict.'.format(k))
                    ignored = True
                    break
            if not ignored:
                new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def load(self, module):
        if module['device'] == 'offline':
            from scepter.modules.utils.import_utils import LazyImportModule
            if (LazyImportModule.get_module_type(('MODELS', module['cfg'].NAME)) or
                    module['cfg'].NAME in MODELS.class_map):
                model = MODELS.build(module['cfg'], logger=self.logger).eval()
            elif (LazyImportModule.get_module_type(('BACKBONES', module['cfg'].NAME)) or
                    module['cfg'].NAME in BACKBONES.class_map):
                model = BACKBONES.build(module['cfg'],
                                        logger=self.logger).eval()
            elif (LazyImportModule.get_module_type(('EMBEDDERS', module['cfg'].NAME)) or
                    module['cfg'].NAME in EMBEDDERS.class_map):
                model = EMBEDDERS.build(module['cfg'],
                                        logger=self.logger).eval()
            else:
                raise NotImplementedError
            if 'DTYPE' in module['cfg'] and module['cfg']['DTYPE'] is not None:
                model = model.to(getattr(torch, module['cfg'].DTYPE))
            if module['cfg'].get('RELOAD_MODEL', None):
                self.init_from_ckpt(module['cfg'].RELOAD_MODEL, model)
            module['model'] = model
            module['device'] = 'cpu'
        if module['device'] == 'cpu':
            module['device'] = we.device_id
            module['model'] = module['model'].to(we.device_id)
        return module

    def unload(self, module):
        if module is None:
            return module
        mem = get_available_memory()
        free_mem = int(mem['available'] / (1024**2))
        total_mem = int(mem['total'] / (1024**2))
        if free_mem < 0.5 * total_mem:
            if module['model'] is not None:
                module['model'] = module['model'].to('cpu')
                del module['model']
            module['model'] = None
            module['device'] = 'offline'
            print('delete module')
        else:
            if module['model'] is not None:
                module['model'] = module['model'].to('cpu')
                module['device'] = 'cpu'
            else:
                module['device'] = 'offline'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return module

    def dynamic_load(self, module=None, name=''):
        self.logger.info('Loading {} model'.format(name))
        if name == 'all':
            for subname in self.loaded_model_name:
                self.loaded_model[subname] = self.dynamic_load(
                    getattr(self, subname), subname)
        elif name in self.loaded_model_name:
            if name in self.loaded_model:
                if module['cfg'] != self.loaded_model[name]['cfg']:
                    self.unload(self.loaded_model[name])
                    module = self.load(module)
                    self.loaded_model[name] = module
                    return module
                elif module['device'] == 'cpu' or module['device'] == 'offline':
                    module = self.load(module)
                    return module
                else:
                    return module
            else:
                module = self.load(module)
                self.loaded_model[name] = module
                return module
        else:
            return self.load(module)

    def dynamic_unload(self, module=None, name='', skip_loaded=False):
        self.logger.info('Unloading {} model'.format(name))
        if name == 'all':
            for name, module in self.loaded_model.items():
                module = self.unload(self.loaded_model[name])
                self.loaded_model[name] = module
        elif name in self.loaded_model_name:
            if name in self.loaded_model:
                if not skip_loaded:
                    module = self.unload(self.loaded_model[name])
                    self.loaded_model[name] = module
            else:
                self.unload(module)
        else:
            self.unload(module)

    def load_default(self, cfg):
        module_paras = {}
        if cfg is not None:
            self.paras = cfg.PARAS
            self.input_cfg = {k.lower(): v for k, v in cfg.INPUT.items()}
            self.input = {k.lower(): dict(v).get('DEFAULT', None) if isinstance(v, (dict, OrderedDict, Config)) else v for k, v in cfg.INPUT.items()}
            self.output = {k.lower(): v for k, v in cfg.OUTPUT.items()}
            module_paras = cfg.MODULES_PARAS
        return module_paras

    def load_image(self, image, num_samples=1):
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, Image):
            pass
        elif isinstance(image, Image):
            pass

    def get_function_info(self, module, function_name=None):
        all_function = module['function_info']
        if function_name in all_function:
            return function_name, all_function[function_name]['dtype']
        if function_name is None and len(all_function) == 1:
            for k, v in all_function.items():
                return k, v['dtype']

    @torch.no_grad()
    def __call__(self,
                 input,
                 **kwargs):
        return

def build_inference(cfg, registry, logger=None, *args, **kwargs):
    """ After build model, load pretrained model if exists key `pretrain`.

    pretrain (str, dict): Describes how to load pretrained model.
        str, treat pretrain as model path;
        dict: should contains key `path`, and other parameters token by function load_pretrained();
    """
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    model = build_from_config(cfg, registry, logger=logger, *args, **kwargs)
    return model

# reigister cls for diffusion.
INFERENCES = Registry('INFERENCE', build_func=build_inference)
