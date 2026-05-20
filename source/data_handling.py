import os
from itertools import combinations
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import re
import json
import unicodedata

from utils import get_id_to_narrative_dict

DATASET_TYPES = {"polynarrative": "multilabel", "CARDS": "multiclass"}

# The text pre-processing functions below are from the CARDS training notebook
# (https://github.com/traviscoan/cards/blob/master/fit/roberta/cards_training.ipynb)


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def strip_underscores(text):
    return re.sub(r'_+', ' ', text)


def remove_multiple_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


def denoise_text(ex):
    text = ex["text"]
    if text is None:
        text = ""
    text = remove_between_square_brackets(text)
    text = remove_non_ascii(text)
    text = strip_underscores(text)
    text = remove_multiple_spaces(text)
    return {"text": text.strip()}


def load_polynarrative(
        home_path,
        partitions,
        languages=["HI", "EN", "BG", "PT"],
        narrative_to_id=None,
        include_language_column=False,
        exclude_other=False,
        other_vs_rest=False):
    # Remove CC labels and split into lists
    def format_labels(ex):
        if "subnarrative" in ex:
            return {k: ex[k].replace("CC: ", "").split(";") for k in {"narrative", "subnarrative"}}

    def add_binary_labels(ex):
        if "subnarrative" in ex:
            c = ex["subnarrative"]
            return {"subnarrative": ("0" if c == ['Other'] else "1")}

    def map_narrative_to_id(ex):
        if "subnarrative" in ex:
            narrative_ids = [narrative_to_id[sn] for sn in ex["subnarrative"]]
            return {"narrative": [n_id.split("_")[0] for n_id in narrative_ids], "subnarrative": narrative_ids}

    all_data = dict()
    for partition in partitions:
        partition_data = []
        for lang in languages:
            labels_f = os.path.join(home_path, partition, lang, "subtask-2-annotations.txt")
            if partition == "train":
                docs_dir = os.path.join(home_path, partition, lang, "raw-documents")
            else:
                docs_dir = os.path.join(home_path, partition, lang, "subtask-2-documents")
            if os.path.exists(labels_f):
                dataset = Dataset.from_csv(
                    labels_f,
                    delimiter="\t",
                    column_names=["id", "narrative", "subnarrative"]
                )
                dataset = dataset.filter(lambda ex: not ex["subnarrative"].startswith("URW"))
            else:
                dataset = Dataset.from_dict({"id": os.listdir(docs_dir)})
                dataset = dataset.filter(lambda ex: not ("URW" in ex["id"] or "RU" in ex["id"]))

            # Add the documents to the dataframe
            def add_text(ex):
                with open(os.path.join(docs_dir, ex["id"])) as f:
                    return {"text": f.read()}

            dataset = dataset.map(add_text)  # .map(denoise_text)
            if include_language_column:
                dataset = dataset.map(lambda ex: {"language": lang})
            partition_data.append(dataset)

        dataset = concatenate_datasets(partition_data)
        all_data[partition] = dataset

    all_data = DatasetDict(all_data)
    all_data.set_format("torch")
    all_data = all_data.map(format_labels)
    if exclude_other:
        all_data = all_data.filter(lambda ex: ex["subnarrative"] != ['Other'])
    if other_vs_rest:
        all_data = all_data.map(add_binary_labels)
    elif narrative_to_id is not None:
        all_data = all_data.map(map_narrative_to_id)
    return all_data


def load_CARDS(data_dir, partitions, exclude_other=False, other_vs_rest=False):
    def add_narrative_cols(ex):
        if ex["claim"] == "0_0":
            narrative, subnarrative = "0", "0"
        else:
            narrative = ex["claim"].split("_")[0]
            subnarrative = ex["claim"]
        return {"narrative": narrative, "subnarrative": subnarrative}

    def add_binary_labels(ex):
        if ex["claim"] == "0_0":
            label = "0"
        else:
            label = "1"
        return {"subnarrative": label}

    partition_mapping = {"training": "train", "validation": "dev", "test": "test"}
    dataset = load_dataset(
        "csv",
        data_files={partition_mapping.get(p, p): os.path.join(data_dir, f"{p}.csv") for p in partitions},
        sep="\t",
        column_names=["text", "claim"])
    splits = dataset["train"].train_test_split(train_size=0.9, seed=42)
    dataset = DatasetDict({"train": splits["train"], "dev": splits["test"], "test": dataset["test"]})

    if exclude_other:
        dataset = dataset.filter(lambda ex: ex["claim"] != "0_0")

    if other_vs_rest:
        dataset = dataset.map(add_binary_labels)
    else:
        dataset = dataset.map(add_narrative_cols)

    dataset = dataset.map(denoise_text)
    dataset.set_format(type="torch")
    return dataset


def load_data(
        dataset_name,
        exclude_other,
        other_vs_rest,
        CARDS_path="CC-denial-resources",
        polynarrative_path="polynarrative",
        taxonomies_path="taxonomies.json",
        partitions=None,
        return_torch_datasets=True):
    dataset_type = DATASET_TYPES.get(dataset_name)
    if other_vs_rest:
        dataset_type = "multiclass"
    if partitions is None:
        if dataset_name == "CARDS":
            partitions = {"training", "test"}
        elif dataset_name == "polynarrative":
            partitions = {"train", "dev"}

    with open(taxonomies_path) as f:
        taxonomies = json.load(f)
    taxonomy_dict = taxonomies[dataset_name]

    id_to_narrative = get_id_to_narrative_dict(taxonomy_dict, 1, dataset_name, convert_to_num=True)
    narrative_to_id = {i: n for n, i in id_to_narrative.items()}

    if dataset_name == "polynarrative":
        dataset = load_polynarrative(
            polynarrative_path,
            partitions,
            narrative_to_id=narrative_to_id,
            include_language_column=True,
            exclude_other=exclude_other,
            other_vs_rest=other_vs_rest)

    elif dataset_name == "CARDS":
        dataset = load_CARDS(
            CARDS_path,
            partitions,
            exclude_other=exclude_other,
            other_vs_rest=other_vs_rest)

    id_to_class = get_id_to_class(dataset, dataset_type)
    class_to_id = {c: i for i, c in id_to_class.items()}

    if return_torch_datasets:
        dataset_test = None  # only defined for some datasets

        # For models implemented in torch, use these datasets
        if dataset_type == "multilabel":
            dataset_train = FlattenedMultiLabelDataset(dataset["train"], narrative_to_class=id_to_class)
            dataset_dev = FlattenedMultiLabelDataset(dataset["dev"], narrative_to_class=id_to_class)
            if dataset_name != "polynarrative":
                dataset_test = FlattenedMultiLabelDataset(dataset["test"], narrative_to_class=id_to_class)

        elif dataset_type == "multiclass":
            dataset_train = FlattenedMultiClassDataset(dataset["train"], narrative_to_class=id_to_class)
            dataset_dev = FlattenedMultiClassDataset(dataset["dev"], narrative_to_class=id_to_class)
            if dataset_name != "polynarrative":
                dataset_test = FlattenedMultiClassDataset(dataset["test"], narrative_to_class=id_to_class)

        return dataset, (dataset_train, dataset_dev, dataset_test), id_to_class, class_to_id

    else:
        return dataset, id_to_class, class_to_id


def get_id_to_class(dataset, dataset_type):
    # Find all unique labels
    unique = set()
    if dataset_type == "multilabel":
        for k in dataset.keys():
            for ns in dataset[k]["subnarrative"]:
                for n in ns:
                    unique.add(n)
    else:
        for k in dataset.keys():
            for n in dataset[k]["subnarrative"]:
                unique.add(n)

    id_to_class = {i: c for c, i in enumerate(sorted(unique, key=lambda i: list(map(int, i.split("_")))))}
    return id_to_class


class FlattenedMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, narrative_to_class=None, unique=None):
        if narrative_to_class is not None:
            self.narrative_to_class = narrative_to_class
        elif unique is None:
            # Find the unique narratives:
            unique = set()
            for ex in dataset["subnarrative"]:
                unique.add(ex)
            self.narrative_to_class = {n: i for i, n in enumerate(sorted(unique))}
        self.class_to_narrative = {i: n for n, i in self.narrative_to_class.items()}
        self.labels = set(self.narrative_to_class.keys())
        self.num_labels = len(self.labels)

        def map_narrative_to_class(ex):
            return {"label": torch.tensor(self.narrative_to_class[ex["subnarrative"]])}
        dataset = dataset.map(map_narrative_to_class)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].get("text"), self.dataset[idx].get("label")


class FlattenedMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, narrative_to_class=None, unique=None):
        if narrative_to_class is not None:
            self.narrative_to_class = narrative_to_class
        elif unique is None:
            # Find the unique narratives:
            unique = set()
            for ex in dataset["subnarrative"]:
                for n in ex:
                    unique.add(n)
            self.narrative_to_class = {n: i for i, n in enumerate(sorted(unique))}
        self.class_to_narrative = {i: n for n, i in self.narrative_to_class.items()}
        self.labels = set(self.narrative_to_class.keys())
        self.num_labels = len(self.labels)

        def label_set_to_multihot(ex):
            return {"label": torch.zeros(self.num_labels).index_fill_(
                0,
                torch.tensor([self.narrative_to_class[n] for n in set(ex["subnarrative"])]),
                1)}
        dataset = dataset.map(label_set_to_multihot)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].get("text"), self.dataset[idx].get("label")


class ChunkBatchSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        self._get_batches()
        self.length = len(self.batches)

    def _get_batches(self):
        indices = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = indices.tolist()
        batches = []
        batch = []
        total_length = 0
        for i in indices:
            length = self.data[i]["input_ids"].shape[0]  # The amount of chunks in the example
            if total_length + length <= self.batch_size:  # Add the index of the item if it fits
                batch.append(i)
                total_length += length
            else:  # If not, check if we have a batch
                if batch:
                    batches.append(batch)
                    if length <= self.batch_size:
                        batch = [i]
                        total_length = length
                    else:
                        batch = []
                        total_length = 0
        # Yield the final batch
        if batch:
            batches.append(batch)
        self.batches = batches

    def __iter__(self):
        if self.shuffle:
            self._get_batches()
            self.length = len(self.batches)
        yield from self.batches

    def __len__(self):
        return self.length


def custom_collate(batch):
    enc = dict()
    enc["lengths"] = [b["attention_mask"].shape[0] for b in batch]
    for k in {"input_ids", "attention_mask"}:
        enc[k] = torch.concatenate([b[k] for b in batch])
    enc["labels"] = torch.stack([batch[i]["labels"] for i in range(len(batch))])
    return enc


def get_custom_batched_tokenized(dataset, split, tokenizer, batch_size, max_length, overlap_proportion=0.4):
    stride_tokens = int(max_length * overlap_proportion)

    def tokenize_with_sliding_window(ex):
        # return_overflowing_tokens splits long docs into multiple chunks
        enc = tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_length,
            stride=stride_tokens,
            return_overflowing_tokens=True,
            padding="max_length",
        )
        enc.pop("overflow_to_sample_mapping")
        enc["labels"] = ex["labels"]
        return enc

    tokenized = dataset[split].map(tokenize_with_sliding_window, batched=False,
                                   remove_columns=dataset[split].column_names)

    sampler = ChunkBatchSampler(tok, batch_size, shuffle=False)

    dataloader_dev = torch.utils.data.DataLoader(tokenized, batch_sampler=sampler, collate_fn=custom_collate)
    return dataloader_dev


# Create contrastive labels
def get_different(ds, column, sample_n, rng, value=[0, 0]):
    grouped = {sn: g for sn, g in ds.groupby(column)}
    sorted_keys = sorted(grouped.keys())
    different = []
    if sample_n is None:
        sample_n = min(map(len, grouped.values()))
    for k1, k2 in combinations(sorted_keys, 2):
        g1, g2 = grouped[k1], grouped[k2]
        local_indices1 = np.arange(len(g1))
        local_indices2 = np.arange(len(g2))
        all_pairs_size = len(g1) * len(g2)
        if sample_n < all_pairs_size:
            selection = rng.choice(np.arange(all_pairs_size), size=sample_n, replace=False)
            _, cs = np.unique(selection, return_counts=True)
            counts, count_counts = np.unique(cs, return_counts=True)
            assert (counts == np.array([1])).all(), counts
        else:
            selection = np.arange(all_pairs_size)
        select1 = [s // len(g2) for s in selection]
        select2 = [s % len(g2) for s in selection]
        for s1, s2 in zip(select1, select2):
            e1, e2 = g1.iloc[local_indices1[s1]], g2.iloc[local_indices2[s2]]
            different.append({"text1": e1["text"], "text2": e2["text"], "label": value})
            assert e1[column] != e2[column]
    return different


def get_same(ds, column, sample_n, rng, value=[1, 1], other_equals_other=True):
    same = []
    grouped = {sn: g for sn, g in ds.groupby(column)}
    if sample_n is None:
        sample_n = min(map(lambda g: math.floor(len(g)*(len(g)-1) / 2), grouped.values()))
    if other_equals_other:
        cats = grouped.keys()
    else:
        cats = sorted(set(grouped.keys()).difference({"0_0", "0"}))
    for k in cats:
        g = grouped[k]
        N = len(g)

        n_all_combs = N * (N - 1) // 2
        selection = rng.choice(np.arange(n_all_combs), size=sample_n, replace=False)
        _, cs = np.unique(selection, return_counts=True)
        counts, count_counts = np.unique(cs, return_counts=True)
        assert (counts == np.array([1])).all(), counts

        # Backwards mapping from index in combination array to the combination itself
        def index_to_pair(i, N):
            # the first in the pair
            c1 = np.floor(N - 0.5 - np.sqrt(np.power(N, 2) - N + 0.25 - 2 * i))
            sum_c1_indices = c1 * (2 * N - 1 - c1) / 2
            # the second in the pair
            c2 = i - sum_c1_indices + c1 + 1
            return np.stack([c1, c2]).astype(int).transpose()

        pairs = index_to_pair(selection, N)
        for s1, s2 in pairs:
            e1, e2 = g.iloc[s1], g.iloc[s2]
            same.append({"text1": e1["text"], "text2": e2["text"], "label": value})
            assert e1[column] == e2[column]
    return same


def get_semi_similar_counts(ds):
    grouped = {n: g for n, g in ds.groupby("narrative")}
    combos = 0
    candidate = None
    for k in grouped.keys():
        counts = [len(g) for _, g in grouped[k].groupby("subnarrative")]
        # Only if there are multiple subnarratives within the narrative can we get combinations
        if len(counts) > 1:
            least, second_least = sorted(counts)[:2]
            n_pairs = least * second_least
            if not candidate or n_pairs < candidate:
                candidate = n_pairs
            combos += len(counts) * (len(counts) - 1) / 2
    return candidate, combos


def get_semi_similar(ds, sample_n, rng, value=[1, 0]):
    semi_similar = []
    grouped = {n: g for n, g in ds.groupby("narrative")}
    for k in grouped.keys():
        different = get_different(grouped[k], "subnarrative", sample_n, rng, value=value)
        semi_similar.extend(different)
    return semi_similar


def get_flattened_contrastive_dataset(dataset, seed=42, neg_pos_ratio=1, desired_total_count=None):
    """
        Takes a dataset with 'text' and 'subnarrative' columns and returns a dataset with 'text1', 'text2' and 'label'
        columns, where 'label'=1 for 'same narrative' and 0 for 'different narratives'.

        The resulting dataset gets a 50-50 balance between 0 and 1, as well as class balance within the two categories.
    """
    rng = np.random.default_rng(seed=seed)

    # 1. Get a good distribution
    num_labels = len(np.unique(dataset["subnarrative"]))
    counts = [len(g) for sn, g in dataset.groupby("subnarrative")]

    least_two = sorted(counts)[:2]
    least, second_least = least_two
    same_count_balanced = least * (least - 1) / 2 * num_labels
    diff_count_balanced = least * second_least * num_labels * (num_labels - 1) / 2

    if same_count_balanced >= neg_pos_ratio * diff_count_balanced:
        same_count_balanced = neg_pos_ratio * diff_count_balanced
    else:
        diff_count_balanced = same_count_balanced // neg_pos_ratio
    total_count = same_count_balanced + diff_count_balanced

    if desired_total_count is None:
        desired_total_count = total_count

    same_prop = 1 / (1 + neg_pos_ratio)
    diff_prop = 1 - same_prop

    n_per_same = int(same_prop * desired_total_count / num_labels)
    n_per_diff = int(diff_prop * desired_total_count / ((num_labels * (num_labels - 1)) / 2))

    # 2. Do the sampling
    different = get_different(dataset, "subnarrative", n_per_diff, rng, value=0)
    same = get_same(dataset, "subnarrative", n_per_same, rng, value=1)
    combined = Dataset.from_list(same + different)
    combined.set_format(type="torch")
    return combined


def get_hierarchical_contrastive_dataset(dataset, seed=42, desired_total_count=None):
    rng = np.random.default_rng(seed=seed)

    # 1. Get a good distribution
    num_labels = len(np.unique(dataset["subnarrative"]))
    counts = [len(g) for _, g in dataset.groupby("subnarrative")]

    least_two = sorted(counts)[:2]
    least, second_least = least_two
    same_count_balanced = least * (least - 1) / 2 * num_labels
    diff_count_balanced = least * second_least * num_labels * (num_labels - 1) / 2

    smallest_semi_similar, n_semi_similar_combos = get_semi_similar_counts(dataset)
    semi_similar_count_balanced = smallest_semi_similar * n_semi_similar_combos

    total_count = 3 * min(same_count_balanced, diff_count_balanced, semi_similar_count_balanced)

    if desired_total_count is None:
        desired_total_count = total_count

    n_per_same = int(desired_total_count / (3 * num_labels))
    n_per_diff = int(desired_total_count / (3 * (num_labels * (num_labels - 1) / 2)))
    n_per_semi = int(desired_total_count / (3 * n_semi_similar_combos))

    # 2. Do the sampling
    different = get_different(dataset, "subnarrative", n_per_diff, rng, value=0)
    same = get_same(dataset, "subnarrative", n_per_same, rng, value=1)
    semi_similar = get_semi_similar(dataset, n_per_semi, rng, value=2)

    combined = Dataset.from_list(same + different + semi_similar)
    combined.set_format(type="torch")
    return combined
