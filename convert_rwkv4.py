# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/6 14:44
from deep_training.nlp.models.rwkv4.convert_rwkv_checkpoint_to_hf import convert_rmkv_checkpoint_to_hf_format

if __name__ == '__main__':

    # repo_id = None
    # checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023.pth'
    # output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-169m'
    # size = '169M'

    repo_id = None
    checkpoint_file = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066.pth'
    output_dir = '/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-pile-430m'
    size = '430M'


    size = size.upper()
    convert_rmkv_checkpoint_to_hf_format(repo_id,checkpoint_file=checkpoint_file,output_dir=output_dir,size=size)
