# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import glob
import sys
import os
from functools import cache

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import copy
import json
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig
from data_processer import DataStrategy, TokenIdsMaker, build_template
from config import *
from aigc_zoo.model_zoo.rwkv4.llm_model import RwkvConfig,set_model_profile,PetlArguments,LoraConfig,PromptArguments
from aigc_zoo.model_zoo.rwkv4.rwkv4_tokenizer import RWKVTokenizer

data_conf = {
    'strategy': DataStrategy.tunction,  # 数据策略选项
    DataStrategy.tunction: {
        'sup': True, # 是否监督模式
    },

    DataStrategy.slidding: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
        'sup': True, # 是否监督模式
        "src_max_length": train_info_args['max_seq_length'] - 10,
        "dst_max_length": None,
    }

}

def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)
        assert data_conf[DataStrategy.slidding]['stride'] > 0

    def load_tokenizer_and_config(self,*args,**kwargs):
        if 'config_class_name' not in kwargs:
            kwargs['config_class_name'] = RwkvConfig
        if 'tokenizer_class_name' not in kwargs:
            if os.path.basename(self.model_args.model_name_or_path).lower().find('world') != -1:
                kwargs['tokenizer_class_name'] = RWKVTokenizer
                tokenizer_kwargs = kwargs.get('tokenizer_kwargs',{})
                kwargs['tokenizer_kwargs'] = tokenizer_kwargs
                tokenizer_kwargs['bos_token_id'] = 0
                tokenizer_kwargs['eos_token_id'] = 0
                tokenizer_kwargs['pad_token_id'] = 1
                tokenizer_kwargs['sep_token_id'] = None

        ret = super().load_tokenizer_and_config(*args,**kwargs)
        self._preprocess_tokenizer_config()
        return ret

    def _preprocess_tokenizer_config(self):
        # model_args = self.model_args
        tokenizer = self.tokenizer
        config = self.config
        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id
        assert config.decoder_start_token_id == config.bos_token_id


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.tunction:
            ds = TokenIdsMaker.tunction(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                        **data_conf[strategy])
        elif strategy == DataStrategy.slidding:
            ds = TokenIdsMaker.slidding(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                        **data_conf[strategy])

        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_paragraph(self, lines):
        D = [ ]
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd[ 'paragraph' ]
            if line_id < 10:
                print(paragraph)

            paragraph = [ (session.get("role", ""), preprocess(session[ 'q' ]),
                           preprocess('\n'.join(session[ 'a' ])) if isinstance(session[ 'a' ], list) else preprocess(
                               session[ 'a' ]))
                          for session in paragraph ]
            sub = [ ]
            # 自行做模板
            for (role, q, a) in paragraph:
                # 不是system prompt  answer 必须存在
                if role != "system":
                    assert len(a), ValueError('answer cannot empty')
                sub.append((role, q, a))
            D.append(copy.deepcopy(sub))
            sub.clear()
        return D

    def _get_messages(self, lines):
        D = [ ]
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd[ 'conversations' ]
            if line_id < 10:
                print(conversations)

            cid = 0
            sub = [ ]
            while cid < len(conversations):
                m = conversations[ cid ]
                cid += 1
                role = m[ "from" ]
                q = preprocess(m[ "value" ])
                if role == "system":
                    a = ""
                    sub.append((role, q, a))
                    continue
                assert role in [ 'user', 'observation', 'function' ]
                m = conversations[ cid ]
                cid += 1
                assert m[ "from" ] == "assistant"
                a = preprocess(m[ "value" ])
                assert len(a), ValueError('answer cannot empty')
                sub.append((role, q, a))
            D.append(sub)
        return D
        # 读取文件

    def on_get_corpus(self, files: typing.List, mode: str):
        D = [ ]
        files = sum([ glob.glob(file) for file in files ], [ ])
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            is_new = False
            if len(lines) > 0:
                is_new = 'conversations' in json.loads(lines[ 0 ])
            if is_new:
                D.extend(self._get_messages(lines))
            else:
                D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "attention_mask": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

        # 记录缓存文件
        with open(os.path.join(data_args.output_dir, 'intermediate_file_index.json'), mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps({
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }, ensure_ascii=False))

    @cache
    def load_dataset_files(self):
        data_args = self.data_args

        if not data_args.convert_file:
            return {
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }

        filename = os.path.join(data_args.output_dir, 'intermediate_file_index.json')
        assert os.path.exists(filename), 'make you dataset firstly'
        with open(filename, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())


if __name__ == '__main__':

    if global_args["trainer_backend"] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args["trainer_backend"] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
        model_args, training_args, data_args, _, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break