# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import logging
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import ModelCheckpointEx
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config, global_args
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, PetlArguments, LoraConfig, PromptArguments,RwkvConfig,set_model_profile

assert global_args["trainer_backend"] == "pl"

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments,PromptArguments))
    model_args, training_args, data_args, lora_args,prompt_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    prompt_args = prompt_args.config

    output_weight_dir = './best_ckpt'







    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args['config_merge']:
        config_kwargs.update(global_args['config_merge'])

    config: RwkvConfig
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)

    is_bf16_supported = torch.cuda.is_bf16_supported()
    precision = global_args["precision"]
    if precision == "auto":
        # 精度 根据实际情况做调整
        if is_bf16_supported:
            precision = 'bf16'
        else:
            precision = '16'

        if global_args[ "quantization_config" ] is not None and global_args[ "quantization_config" ].load_in_8bit:
            precision = "32"

    if precision.startswith('16'):
        RWKV_FLOAT_MODE = '16'
    elif precision.startswith('bf16'):
        RWKV_FLOAT_MODE = 'bf16'
    else:
        RWKV_FLOAT_MODE = '32'
        # 加载cuda_core
    set_model_profile(RWKV_T_MAX=config.ctx_len, RWKV_FLOAT_MODE=RWKV_FLOAT_MODE)

    # 训练数据最大长度限制
    assert data_args.train_max_seq_length <= config.ctx_len

    dataHelper.make_dataset_all()


    deepspeed_config = get_deepspeed_config(precision)
    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )
    checkpoint_callback = ModelCheckpointEx(
        # monitor='loss',
        dirpath=output_weight_dir,
        save_weights_only=True,
        save_last=True,
        # every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
        every_n_epochs=1,
        lora_args=lora_args,
        prompt_args=prompt_args,
        # monitor="loss",mode = "min", save_top_k = 10 按loss存储10个模型
        monitor="step", mode="max",
        save_top_k=10,  # 按步存储最后10个模型
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy=strategy,
        # lora int8 precision='32'
        precision=precision,# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )

    transformer_args = dict(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args, prompt_args=prompt_args,
                             quantization_config=global_args["quantization_config"],
                             device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                             torch_dtype=torch.float16,
                             new_num_tokens=len(tokenizer),  # 可能扩充词
                             )

    if transformer_args["quantization_config"] is None:
        transformer_args.pop("device_map")

    pl_model = MyTransformer(**transformer_args)
    config.save_pretrained(output_weight_dir)

    # 加载sft权重
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    pl_model = pl_model.float()

    def dataset_loader_filter_fn(dataset):
        print('*' * 30, 'total', len(dataset))
        return dataset


    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=data_args.data_backend == 'record',
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
        num_workers=0
    )

    if train_datasets is not None:
        trainer.fit(pl_model, train_dataloaders=train_datasets)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
