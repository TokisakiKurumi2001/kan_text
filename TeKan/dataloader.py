from typing import List
from collections.abc import Mapping

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

class TeKANDataLoader:
    def __init__(
        self,
        tok_pretrained_ck: str,
        max_length: int,
        valid_ratio: float,
        num_train_sample: int = -1,
    ):
        dataset = load_dataset('imdb')
        if num_train_sample != -1:
            dataset['train'] = dataset['train'].select(range(num_train_sample))
            dataset['test'] = dataset['test'].select(range(num_train_sample))
        dataset.pop('unsupervised')
        self.tokenizer = AutoTokenizer.from_pretrained(tok_pretrained_ck)
        self.max_length = max_length
        tokenized_datasets = dataset.map(
            self.__tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
        )
        dataset_split = tokenized_datasets['train'].train_test_split(test_size=valid_ratio)
        dataset_split['validation'] = dataset_split.pop('test')
        dataset_split['test'] = tokenized_datasets['test']
        self.dataset = dataset_split

    def __tokenize_function(self, examples):
        result = self.tokenizer(examples["text"], max_length=self.max_length, truncation=True, padding="max_length")
        result['labels'] = examples['label']
        return result
    
    def collator_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size: int = 16, types: List[str] = ["train", "test", "validation"]):
        res = []
        for t in types:
            if t == "train":
                shuffle = True
            else:
                shuffle = False
            res.append(
                DataLoader(
                    self.dataset[t],
                    batch_size=batch_size,
                    collate_fn=self.collator_fn,
                    num_workers=2,
                    shuffle=shuffle,
                )
            )
        return res