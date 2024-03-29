# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config,build_template
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, RwkvConfig,set_model_profile
from aigc_zoo.utils.rwkv4_generate import Generate

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    # 可以自行修改 RWKV_T_MAX  推理最大长度
    set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='')

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    

    config = RwkvConfig.from_pretrained('./best_ckpt')
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=torch.float16)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = './best_ckpt/last.ckpt'
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]
    for input in text_list:
        query = build_template(input)
        response = Generate.generate(model, query=query, tokenizer=tokenizer, max_length=512,
                                     eos_token_id=config.eos_token_id,
                                     pad_token_id=config.eos_token_id,
                                     do_sample=True, top_p=0.85, temperature=1.0, )
        print('input',input)
        print('output',response)