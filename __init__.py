# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from .ace_plus_fft_node import ACEPlusFFTLoader, ACEPlusFFTConditioning, AcePlusFFTProcessor

NODE_MAPPINGS = {
    'ACEPlusLoader': ('ACEPlusFFTLoader~', ACEPlusFFTLoader),
    'ACEPlusConditioning': ('ACEPlusFFTConditioning~', ACEPlusFFTConditioning),
    'ACEPlusFFTProcessor': ('ACEPlusFFTProcessor~', AcePlusFFTProcessor)
}

NODE_CLASS_MAPPINGS = {k: v[1] for k, v in NODE_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v[0] for k, v in NODE_MAPPINGS.items()}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
