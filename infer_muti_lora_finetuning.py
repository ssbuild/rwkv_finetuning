# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import os

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, PetlArguments, \
    RwkvConfig,set_model_profile,PetlModel
from aigc_zoo.utils.llm_generate import Generate

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    # 可以自行修改 RWKV_T_MAX  推理最大长度
    set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='')

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=RwkvConfig)
    

    # 一般根据时间排序选最新的权重文件夹
    ckpt_dir = './best_ckpt/last'

    config = RwkvConfig.from_pretrained(ckpt_dir)
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )

    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()


    # backbone model replaced PetlModel
    lora_model: PetlModel = pl_model.backbone

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]

    # 基准模型推理
    with lora_model.disable_adapter():
        for input in text_list:
            # lora_model 调用子对象方法
            response, history = Generate.chat(lora_model, query=input, tokenizer=tokenizer, max_length=512,
                                              eos_token_id=config.eos_token_id,
                                              pad_token_id=tokenizer.eos_token_id,
                                              do_sample=False, top_p=0.7, temperature=0.95, )
            print('input', input)
            print('output', response)

    lora_model.set_adapter(adapter_name='default')

    for input in text_list:
        # lora_model 调用子对象方法
        response, history = Generate.chat(lora_model, query=input, tokenizer=tokenizer, max_length=512,
                                          eos_token_id=config.eos_token_id,
                                          pad_token_id=tokenizer.eos_token_id,
                                          do_sample=False, top_p=0.7, temperature=0.95, )
        print('input', input)
        print('output', response)
