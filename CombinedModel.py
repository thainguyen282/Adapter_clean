from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import os
from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers import set_seed
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from utils.prompter import Prompter
import transformers
from utils.prompter import Prompter
import gc
import copy
import json

def main():
    prompter = Prompter("alpaca")
    tokenizer = AutoTokenizer.from_pretrained("/mmfs1/project/phan/codellama/model/")
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_size = "left"
    class MergedConfig(PretrainedConfig):
        model_type = "merged"

        def __init__(
            self, 
            enc_dec_model: Optional[str] = None, 
            enc_num_layers: int = 1, 
            dec_num_layers: int = 4, 
            middle_model: Optional[str] = None, 
            middle_num_layers: Optional[int] = None, 
            enc_config: dict = {}, 
            middle_config: dict = {}, 
            dec_config: dict = {}, 
            adap1_config: dict = {}, 
            adap2_config: dict = {},
            **kwargs, 
        ):
            self.enc_dec_model = enc_dec_model
            self.enc_num_layers = enc_num_layers
            self.dec_num_layers = dec_num_layers 
            self.middle_model = middle_model 
            self.middle_num_layers = middle_num_layers 
            self.enc_config = enc_config 
            self.middle_config = middle_config 
            self.dec_config = dec_config
            self.adap1_config = adap1_config
            self.adap2_config = adap2_config
            super().__init__(**kwargs) 

    class MergedModel(PreTrainedModel):
        config_class = MergedConfig
        
        def __init__(self, config: MergedConfig):
            super().__init__(config)
            self.client = AutoModelForCausalLM.from_pretrained(config.enc_dec_model, **config.enc_config)
            self.encoder = copy.deepcopy(self.client.model)
            self.encoder.layers = self.encoder.layers[:config.enc_num_layers]
            self.mid = AutoModel.from_pretrained(config.middle_model, **config.middle_config)
            # self.decoder = copy.deepcopy(self.client)
            self.decoder = self.client
            self.decoder.model.layers = self.decoder.model.layers[-config.dec_num_layers:]
            self.adapter1 = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, 3754, bias=False).to('cuda:0'), 
                nn.GELU(), 
                nn.Linear(3754, 3924, bias=False).to('cuda:0'), 
                nn.GELU(),
                nn.Linear(3924, self.mid.config.hidden_size, bias=False).to('cuda:0'), 
                self.mid.norm)
            self.adapter2 = nn.Sequential(
                nn.Linear(self.mid.config.hidden_size, 3924, bias=False).to('cuda:0'), 
                nn.GELU(), 
                nn.Linear(3924, 3754, bias=False).to('cuda:0'), 
                nn.GELU(),
                nn.Linear(3754, self.decoder.config.hidden_size, bias=False).to('cuda:0'), 
                self.decoder.model.norm)
            self.act_fn = nn.GELU() 
            for layer in self.adapter1:
                if isinstance(layer, nn.Linear):
                    # Apply Kaiming Uniform initialization to the weight of each Linear layer
                    nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_out", nonlinearity="relu")
                    layer.weight = nn.Parameter(layer.weight.to(torch.bfloat16))
            for layer in self.adapter2:
                if isinstance(layer, nn.Linear):
                    # Apply Kaiming Uniform initialization to the weight of each Linear layer
                    nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_out", nonlinearity="relu")
                    layer.weight = nn.Parameter(layer.weight.to(torch.bfloat16))
            # nn.init.kaiming_uniform_(self.adapter1[0].weight, a=0, mode="fan_out", nonlinearity='relu')
            # nn.init.kaiming_uniform_(self.adapter2[0].weight, a=0, mode="fan_out", nonlinearity='relu')
            # self.adapter1[0].weight = nn.Parameter(self.adapter1[0].weight.to(torch.bfloat16))
            # self.adapter2[0].weight = nn.Parameter(self.adapter2[0].weight.to(torch.bfloat16))
            # self.adapter1[0].weight = nn.Parameter(self.adapter1[0].weight)
            # self.adapter2[0].weight = nn.Parameter(self.adapter2[0].weight)
            gc.collect()
            torch.cuda.empty_cache()

        def forward (
                self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask = attention_mask,
                use_cache = False
            )

            X = self.adapter1(encoder_out.last_hidden_state)
            X = self.act_fn(X)
            
            mid_out = self.mid(
                inputs_embeds=X.to(self.mid.device),
                attention_mask=attention_mask,
                use_cache=False
            )

            X = self.adapter2(mid_out.last_hidden_state)
            X = self.act_fn(X)

            decoder_out = self.decoder(
                inputs_embeds=X.to(self.decoder.device), 
                attention_mask=attention_mask, 
                use_cache=False, 
                labels=labels
            )

            return decoder_out

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            return self.decoder.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        
        def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = False,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
        ):
            state_dict = {}
            # Remove the middle model
            for key in self.state_dict().keys():
                if not key.startswith("mid"):
                    state_dict[key] = self.state_dict()[key]

            kwargs.pop("state_dict", None)
            safe_serialization = False
            # Save the encoder-decoder model
            super().save_pretrained(
                save_directory,
                state_dict=state_dict,
                is_main_process=is_main_process,
                save_function=save_function,
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                variant=variant,
                token=token,
                save_peft_format=save_peft_format,
                **kwargs,
            )


    instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

    def make_chat_prompt(
        task_prompt: str,
        instruction_prefix: str,
        response_prefix: str,
        tokenizer: AutoTokenizer,
    ) -> str:
        # directly return prompt if it does not have a tokenizer.chat_template
        if tokenizer.chat_template is None:
            return task_prompt

        assert instruction_prefix is not None, "Instruction prefix is required!"
        assert response_prefix is not None, "Response prefix is required!"

        task_prompt = f"""\
        {instruction_prefix}
        ```
        {task_prompt.strip()}
        ```
        """
        response = f"""\
        {response_prefix}
        ```python
        {_MAGIC_SPLITTER_}
        ```
        """
        #print(1)
        task_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
        return task_prompt


    cutoff_len = 1500
    train_on_inputs = True
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()
        #result["labels"] = torch.empty_like(result["input_ids"]).copy_(result["input_ids"])

        return result


    template = "{}\n```"
    def generate_and_tokenize_prompt(data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            "",
            #data_point["output"],
        )
        #prompt = prompter.generate_prompt(prompt)
        #tempLen = len(prompt)

        messages = make_chat_prompt(full_prompt[len(system)+2:],instruction_prefix, response_prefix, tokenizer)
        
        #system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        # messages = [
        #     #{"role": "system", "content": "You are an expert Python programmer, and here is your task:"},
        #     {"role": "system", "content": system},
        #     {"role": "user", "content": full_prompt[len(system)+2:]},
        #     #{"role": "assisstant", "content": template.format(data_point["output"])}
        # ]
        
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # print(text)
        #output = template.format(data_point["output"])
        output = data_point["output"]
        messages += output

        # print(messages)

        tokenized_full_prompt = tokenize(messages)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
        
    def update_model(model):
        total_param = 0
        trainable_param = 0
        device = model.device
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
            total_param += parameter.numel()
            trainable_param += parameter.numel()
        for parameter in model.mid.parameters():
            parameter.requires_grad = False
            total_param += parameter.numel()
        for parameter in model.decoder.parameters():
            parameter.requires_grad = False
            total_param += parameter.numel()
            trainable_param += parameter.numel()
        for parameter in model.adapter1.parameters():
            parameter.requires_grad = True
            trainable_param += parameter.numel()
            total_param += parameter.numel()
        for parameter in model.adapter2.parameters():
            parameter.requires_grad = True
            trainable_param += parameter.numel()
            total_param += parameter.numel()

        print(f'Total Parameters: {total_param:,}')
        print(f'Trainable Parameters: {trainable_param:,}')
        print(f'Non-trainable Parameters: {total_param - trainable_param:,}')
        
        # Print percentage of trainable parameters
        print(f'Percentage of trainable parameters: {100 * trainable_param / total_param:.2f}%')


    class MergedModelForCausalLM(MergedModel):
        pass

    AutoConfig.register("merged", MergedConfig)
    AutoModel.register(MergedConfig, MergedModel)
    AutoModelForCausalLM.register(MergedConfig, MergedModelForCausalLM)
    model_path = "/mmfs1/project/phan/tqn/infly/OpenCoder-8B-Instruct"
    middle_path = "/mmfs1/project/phan/codellama/Qwen2.5-Coder-7B-Instruct"
    config = MergedConfig(enc_dec_model=model_path, middle_model=middle_path, enc_config={"device_map": "auto", "torch_dtype": torch.bfloat16}, middle_config={"device_map": "auto", "torch_dtype": torch.bfloat16}, dec_config={"device_map": "auto", "torch_dtype": torch.bfloat16}, adap1_config={"device_map": "auto"}, adap2_config={"device_map": "auto"})
    data = load_from_disk("/mmfs1/project/phan/codellama/datasets/PGCodeTraining100k")
    # model = MergedModelForCausalLM.from_pretrained("/mmfs1/project/phan/tqn/Adapter/finetune_model/MLP_model_copy/checkpoint-9000", config=config) 
    model = MergedModelForCausalLM(config=config)
    update_model(model)

    # val_set_size = 2000
    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )

    #     dataset = DatasetDict({
    #         "train": train_data,
    #         "validation": val_data
    #     })
    
    val_set_size = 2000
    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )

    # train_data.to_json("/mmfs1/project/phan/tqn/data/train.jsonl")
    # val_data.to_json("/mmfs1/project/phan/tqn/data/validation.jsonl")
    
    train_data = load_dataset("json", data_files="/mmfs1/project/phan/tqn/data/train.jsonl")["train"]
    val_data = load_dataset("json", data_files="/mmfs1/project/phan/tqn/data/validation.jsonl")["train"]
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # trainer = transformers.Trainer(
    #     model=model,
    #     tokenizer = tokenizer,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=1,
    #         per_device_eval_batch_size=1,
    #         gradient_accumulation_steps=16,
    #         warmup_steps=100,
    #         num_train_epochs=1,
    #         learning_rate=3e-4,
    #         # learning_rate=1e-4,
    #         # fp16=True,
    #         bf16=True,
    #         logging_steps=100,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         # evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         # 
    #         save_strategy="steps",
    #         eval_steps = 5000 if val_set_size > 0 else None,
    #         save_steps=5000,
    #         output_dir="/mmfs1/project/phan/tqn/Adapter/pretrained_model/model3",
    #         save_total_limit=6,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         report_to="wandb",
    #         run_name= "/mmfs1/project/phan/tqn/Adapter/pretrained_model/model3",
    #         # max_grad_norm=1.0
    #         # save_steps=0,  # Disable checkpoint saving
    #         # save_total_limit=0,
    #     ),
    #     # data_collator=transformers.DataCollatorForSeq2Seq(
    #     #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     # ),
    # )

    trainer = transformers.Trainer(
        model=model,
        tokenizer = tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=3e-4,
            # learning_rate=1e-4,
            # fp16=True,
            bf16=True,
            logging_steps=100,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # 
            # save_strategy="steps",
            eval_steps = 500 if val_set_size > 0 else None,
            save_steps=500,
            output_dir="/mmfs1/project/phan/tqn/Adapter/finetune_model/MLP_model_para_reduce",
            save_total_limit=6,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb",
            run_name= "/mmfs1/project/phan/tqn/Adapter/finetune_model/MLP_model_para_reduce"
            # max_grad_norm=1.0
            # save_steps=0,  # Disable checkpoint saving
            # save_total_limit=0,
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )
    trainer.train()
    # model.save_pretrained("/mmfs1/project/phan/tqn/Adapter/pretrained_model/model3", safe_serialization=False)
    
if __name__ == "__main__":
    main()
