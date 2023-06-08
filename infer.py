# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, Generate, RwkvConfig


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16},config_class_name=RwkvConfig)
    dataHelper.preprocess_tokenizer_config()
    pl_model = MyTransformer(config=config, model_args=model_args)
    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    init_prompt = '''
    Q: 企鹅会飞吗？

    A: 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

    Q: 西瓜是什么

    A: 西瓜是一种常见的水果，是一种多年生蔓生藤本植物。西瓜的果实呈圆形或卵形，通常是绿色的，里面有红色或黄色的肉和很多的籽。西瓜味甜，多吃可以增加水分，是夏季非常受欢迎的水果之一。

    '''

    response, history = Generate.chat(model, query=init_prompt, tokenizer=tokenizer, max_length=512,
                                      eos_token_id=config.eos_token_id,
                                      do_sample=False, top_p=0.7, temperature=0.95, )
    print('input', input)
    print('output', response)

    # text_list = ["写一个诗歌，关于冬天",
    #              "晚上睡不着应该怎么办",
    #              "从南京到上海的路线",
    #              ]
    # for input in text_list:
    #     response, history = Generate.chat(model, query=input, tokenizer=tokenizer, max_length=512,
    #                                       eos_token_id=config.eos_token_id,
    #                                       do_sample=False, top_p=0.7, temperature=0.95, )
    #     print('input', input)
    #     print('output', response)