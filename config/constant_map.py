# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]

MODELS_MAP = {
    'rwkv-4-pile-169m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m',
    },
    'rwkv-4-pile-430m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m',
    },

    'rwkv-4-pileplus-430m': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pileplus-430m',
    },


    'rwkv-4-pile-3b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-3b',
    },
    'rwkv-4-pile-7b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-7b',
    },
    'rwkv-4-pile-14b': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-14b',
    },

    # 中英日语
    "rwkv-4-raven-3b-v12-Eng49%-Chn49%-Jpn1%-Other1%": {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12',
    },
    "rwkv-4-Raven-1B5-v12-Eng98%-Other2%": {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-1b5-v12',
    },

    # world 系列
    "rwkv-4-World-CHNtuned-3B-v1": {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-3B-v1',
        'config_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-3B-v1/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-World-CHNtuned-3B-v1',
    }


}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING


