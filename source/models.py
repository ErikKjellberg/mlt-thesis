from typing import Literal, Sequence, TypedDict, Tuple, List, Union, Any, Annotated
from pydantic import BaseModel, Field
import os
from os import path
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import getpass
import logging
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, pipeline
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from transformers.modeling_outputs import SequenceClassifierOutput

os.environ["LANGSMITH_TRACING"] = "true"

Role = Literal["system", "user"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

NarrativeID = Annotated[str, Field(pattern=r"^(\d+\.([a-z]|Other)|Other)$")]


class NarrativeList(BaseModel):
    narratives: List[NarrativeID]


class Narrative(BaseModel):
    narrative: NarrativeID


class LlamaClassifier:

    def __init__(self,
                 model_name,
                 system_message_template,
                 user_message_template,
                 hf_token=None, max_length: int = 512, batch_size: int = 32, max_time: float = 4.5,
                 temperature: float = 1, sampling=False):
        self.model_name = model_name
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_time = max_time
        self.temperature = temperature
        self.sampling = sampling

        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token is None:
                hf_token = getpass.getpass("Enter HuggingFace token: ")

        login(hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = pipeline("text-generation", model=self.model_name, tokenizer=self.tokenizer, device_map="auto")

        self.eos_tokens = [self.tokenizer.eos_token_id]

        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.add_special_tokens = True
        self.model.tokenizer.add_eos_token = True
        self.model.tokenizer.add_bos_token = True

    def apply_chat_template(self, dataset, system_message, template):
        def apply_chat_template_func(record):
            dialog: Dialog = (Message(role='system', content=system_message),
                              Message(role='user', content=template.format(record['text'])))
            return self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True, tokenize=False)

        return [apply_chat_template_func(row) for row in dataset]

    def extract_prediction(self, answer: str) -> str:
        prediction = answer.split('\n')[-1]
        return prediction.replace('\n', ' ')

    def predict(self,
                dataset,
                taxonomy,
                system_message_template: str = None,
                user_message_template: str = None,
                full_messages=False) -> List[str]:
        if system_message_template is None:
            system_message_template = self.system_message_template
        if user_message_template is None:
            user_message_template = self.user_message_template

        system_message = system_message_template.format(taxonomy)

        dataset = self.apply_chat_template(dataset, system_message, user_message_template)

        predictions = []
        total = len(dataset)
        for i in range(0, total, self.batch_size):
            print(f"{i} out of {total} examples")
            batch = dataset[i:i + self.batch_size]
            answers = self.model(
                batch,
                max_new_tokens=self.max_length,
                forced_eos_token_id=self.eos_tokens,
                max_time=self.max_time * self.batch_size,
                eos_token_id=self.eos_tokens,
                temperature=self.temperature,
                do_sample=self.sampling,
                pad_token_id=self.model.tokenizer.eos_token_id
            )
            if full_messages:
                predictions.extend([a[0]['generated_text'] for a in answers])
            else:
                predictions.extend([self.extract_prediction(a[0]['generated_text']) for a in answers])

        if len(predictions) != total:
            raise ValueError(f"Predictions count ({len(predictions)}) doesn't match input count ({total}).")
        return predictions


class DeepSeekClassifier:

    def __init__(self, model_name_or_path: str, model_provider: str = None, langsmith_key: str = None,
                 provider_key_name: str = "DEEPSEEK_API_KEY", provider_key: str = None, temperature=0, max_tokens=None,
                 timeout=None, max_retries=2, structure: BaseModel = None, system_message_template=None,
                 user_message_template=None, **kwargs):

        self.model_name = model_name_or_path

        if langsmith_key != None:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
        elif not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

        if provider_key != None:
            os.environ[provider_key_name] = provider_key
        elif provider_key_name != None and not os.environ.get(provider_key_name):
            os.environ[provider_key_name] = getpass.getpass(f"Enter API key for {model_provider}: ")

        if model_provider == "deepseek":
            self.llm = ChatDeepSeek(
                model=model_name_or_path,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries
            )
        else:
            self.llm = init_chat_model(model_name_or_path, model_provider=model_provider)

        self.set_structure(structure)

        self.system_message_template = system_message_template
        self.user_message_template = user_message_template

    def set_structure(self, structure):
        if structure is not None:
            self.model = self.llm.with_structured_output(structure)
            self.name = self.model_name + "_" + structure.__name__
        else:
            self.model = self.llm
            self.name = self.model_name

    def predict(self, dataset,
                taxonomy,
                system_message_template=None,
                user_message_template=None,
                response_attribute=None
                ):

        if system_message_template is None:
            system_message_template = self.system_message_template
        if user_message_template is None:
            user_message_template = self.user_message_template

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message_template), ("user", user_message_template)]
        )

        predictions = []

        for i in range(len(dataset)):
            print(f"{i} out of {len(dataset)} examples")
            ex = dataset[i]
            text = ex["text"]

            prompt = prompt_template.invoke({"text": text, "taxonomy": taxonomy})

            try:
                response = self.model.invoke(prompt)
            except:
                logging.error("Could not run chat completion.")
                raise Exception

            if response_attribute is not None:
                response = getattr(response, response_attribute) or response.content
            predictions.append(response)
        return predictions


class RoBERTaClassifierConfig(PretrainedConfig):
    model_type = "roberta_classifier"

    def __init__(self, model_name="roberta_base", n_classes=2, aggregation="mean", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.n_classes = n_classes
        self.aggregation = aggregation


class RoBERTaClassifier(PreTrainedModel):
    config_class = RoBERTaClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.emb_size = self.model.config.hidden_size
        self.aggregation = config.aggregation
        self.attention = nn.Linear(self.emb_size, 1)
        # Below is from the transformers source to replicate their classification head
        self.fc1 = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(self.emb_size, config.n_classes)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        lengths=None,
        return_embeddings=False,
        aggregation=None,
        **kwargs,
    ):
        aggregation = aggregation or self.aggregation
        out = self.model(input_ids,
                         attention_mask=attention_mask,
                         **kwargs,)
        embs = out["last_hidden_state"][:, 0, :]

        # Average across chunks for each example
        if aggregation == "mean":
            embs = torch.stack([s.mean(dim=0) for s in torch.split(embs, tuple(lengths))])
        # Attend over chunks for each example
        elif aggregation == "attention":
            weighted = []
            for s in torch.split(embs, tuple(lengths)):
                attention_scores = self.attention(s).squeeze(-1)
                attention_weights = torch.softmax(attention_scores, dim=0)
                pooled = torch.sum(attention_weights.unsqueeze(-1) * s, dim=0)
                weighted.append(pooled)
            embs = torch.stack(weighted)

        # Below is from the transformers source to replicate their classification head
        x = self.dropout(embs)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_embeddings:
            return SequenceClassifierOutput(logits=x), embs.cpu().numpy()
        return SequenceClassifierOutput(logits=x)
