from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster._unsupervised import silhouette_samples
import bcubed
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils import (
    to_coarse, 
    label_set_to_multihot, 
    logits_to_multihot, 
    logits_to_multihot_allow_no_pred, 
    multihot_to_list_of_classes
)
from data_handling import get_custom_batched_tokenized

# Define additional model performance scores (F1) (copied from CARDS notebook)
def f1_multiclass_macro(labels, preds):
    return f1_score(labels, preds, average='macro')
def f1_multiclass_micro(labels, preds):
    return f1_score(labels, preds, average='micro')
def f1_multiclass_weighted(labels, preds):
    return f1_score(labels, preds, average='weighted')
def f1_class(labels, preds):
    return f1_score(labels, preds, average=None)
def precision(labels, preds):
    return precision_score(labels, preds, average='macro')
def recall(labels, preds):
    return recall_score(labels, preds, average='macro')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_multiclass_macro(labels, preds),
        "f1_micro": f1_multiclass_micro(labels, preds),
        "f1_weighted": f1_multiclass_weighted(labels, preds),
        "precision_macro": precision(labels, preds),
        "recall_macro": recall(labels, preds),
        "f1_per_class": ",".join([f"{s:.4f}" for s in f1_score(labels, preds, average=None)]),
    }

def compute_metrics_multilabel(eval_pred, min_p=0.5):
    logits, labels = eval_pred
    preds = np.array([logits_to_multihot(torch.Tensor(p), min_p=min_p) for p in logits])

    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_samples": f1_score(labels, preds, average="samples"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_per_class": ",".join([f"{s:.4f}" for s in f1_score(labels, preds, average=None)]),
    }


def get_predictions(model, dataloader, dataset_type, min_p=0.5, extract_embeddings=False):
    preds = []
    truth = []
    embs = []
    model.eval()

    with tqdm(dataloader) as pbar:
        for i, data in enumerate(pbar):
            for k in {"input_ids", "attention_mask"}:
                data[k] = data[k].to(model.device)
            labels = data["labels"]
            with autocast():
                if extract_embeddings:
                    classifier_output, e = model(**data, return_embeddings=True)
                    pred = classifier_output.logits
                    embs.append(e)
                else:
                    pred = model(**data).logits

                if dataset_type == "multiclass":
                    pred = pred.argmax(dim=1).cpu().numpy()
                else:
                    pred = [logits_to_multihot(torch.Tensor(p), min_p=min_p).cpu() for p in pred]

                preds.append(list(pred))
                truth.append(list(labels))

    preds = np.concatenate(preds)
    truth = np.concatenate(truth)

    if extract_embeddings:
        embs = np.vstack(embs)
        return truth, preds, embs
    return truth, preds


def get_predictions_double_classifier(
        binary_model,
        narrative_model,
        binary_dataloader,
        narrative_dataloader,
        dataset_type,
        min_p=0.5,
        extract_embeddings=False):
    truth, binary_preds = get_predictions(binary_model, binary_dataloader, "multiclass")
    _, narrative_preds = get_predictions(narrative_model, narrative_dataloader, dataset_type, min_p=min_p,
                                         extract_embeddings=extract_embeddings)
    final_preds = np.where(binary_preds.astype(bool), narrative_preds + 1, 0)
    return truth, final_preds


def get_multiclass_predictions_multilabel(
        dataloader, 
        tokenizer, 
        model, 
        aggregation="mean", 
        min_p=0.5,
        max_length=256, 
        overlap=0.4):
    """
        Use this for evaluating a multiclass (CARDS) model on a multilabel (PolyNarrative) dataset.

        Excepts a dataloader that yields a tuple (inputs, labels), where
            inputs = {
                "input_ids": Tensor[n_chunks, seq_len],
                "attention_mask": Tensor[n_chunks, seq_len],
                "lengths": list of length n_chunks
            }
    """
    stride_tokens = int(max_length * overlap)
    preds = []
    truth = []
    for ex in dataloader:
        # Each example consists of a text and a set of labels
        # tokenized becomes a tensor of shape [n_chunks, max_length]
        tokenized = tokenizer(ex[0],
            truncation=True,
            max_length=max_length,
            stride=stride_tokens,
            padding="max_length",
            return_tensors="pt"
        )
        features = {k:tokenized[k].to("cuda") for k in {"input_ids", "attention_mask"}}
        if aggregation == "mean":
            logits = model(**features, aggregation=aggregation, lengths=[tokenized["input_ids"].shape[0]]).logits
            pred = logits_to_multihot(logits.cpu().long(), min_p=min_p)
        elif aggregation == "union":
            # logits is of shape [n_chunks, max_length]
            logits = model(**features, aggregation="none").logits
            pred_per_chunk = logits_to_multihot_allow_no_pred(logits.cpu().long(), min_p=min_p)
            pred = torch.any(pred_per_chunk, dim=0).int()
            if pred.sum() == 0:
                pred[logits.sum(dim=0).argmax()] = 1
            
        preds.append(pred)
        truth.extend(ex[1])
    return np.array(truth), np.array(preds)


def get_multilabel_predictions_multiclass(dataloader, model, max_length=256, overlap=0.4):
    """
        Use this for evaluating a multi-label model on a multiclass dataset.

        Excepts a dataloader that with already tokenized examples.
    """
    stride_tokens = int(max_length * overlap)
    preds = []
    truth = []
    model.eval()
    for ex in dataloader:
        features = ex | {k:ex[k].to("cuda") for k in {"input_ids", "attention_mask"}}
        logits = model(**features).logits
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        truth.extend(ex["labels"])
    return np.array(truth), np.array(preds)


def get_classes_and_embeddings(model, dataloader, dataset_type, aggregation=None, overlap=0.4):
    """
        Takes an SBERT model and a dataloader, tokenizes examples and returns embeddings and true labels.
    """
    def tokenize_with_sliding_window(tokenizer, text, max_length, overlap):
        # return_overflowing_tokens splits long docs into multiple chunks
        stride_tokens = int(overlap * max_length)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride_tokens,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt"
        )
        return enc

    embs = []
    truth = []
    model.eval()

    with tqdm(dataloader) as pbar:
        for i, data in enumerate(pbar):
            inputs, labels = data
            if dataset_type == "multilabel":
                # For PolyNarrative, we need to split the document into multiple chunks
                encoded = tokenize_with_sliding_window(model.tokenizer, inputs, model.max_seq_length, overlap=overlap)
                chunks = [model.tokenizer.decode(e) for e in encoded["input_ids"]]
                e = model.encode(chunks)
                if aggregation == "mean":
                    e = np.mean(e, axis=0)
            else:
                e = model.encode(inputs)

            truth.append(list(labels))
            embs.append(e)

    truth = np.concatenate(truth)
    if dataset_type == "multiclass":
        embs = np.concatenate(embs)
    elif aggregation == "mean":
        embs = np.array(embs)

    return truth, embs


def get_predictions_sbert_binary_classifier(
        binary_model,
        contrastive_model,
        binary_dataloader,
        contrastive_dataloader,
        clustering_method,
        dataset_type,
        extract_embeddings=False,
        aggregation="mean"
):
    """
        Predicts on the dataset using a binary classifier along with sbert representations, that are clustered.
        clustering_method should be of type np.array -> np.array, returning labels given some embeddings
    """
    truth, binary_preds = get_predictions(binary_model, binary_dataloader, "multiclass")
    truth_2, embs = get_classes_and_embeddings(contrastive_model, contrastive_dataloader, dataset_type,
                                               aggregation=aggregation)
    assert (truth == truth_2).all()

    embs_to_cluster = embs[binary_preds != 0]
    labels = clustering_method(embs_to_cluster)

    if dataset_type == "multiclass":
        final_preds = binary_preds
        final_preds[binary_preds.astype(bool)] = labels + 1
    else:
        unique_labels = set().union(*labels)
        unique_labels = {0}.union({l + 1 for l in unique_labels})
        final_preds = np.zeros((len(truth), len(unique_labels)))
        other = np.zeros((len(unique_labels)))
        other[0] = 1
        final_preds[~binary_preds.astype(bool)] = other
        final_preds[binary_preds.astype(bool)] = np.array([
            label_set_to_multihot(
                {l + 1 for l in ll},
                {l: i for i, l in enumerate(unique_labels)}) for ll in labels])

    assert len(truth) == len(final_preds)
    if extract_embeddings:
        return truth, final_preds, embs
    else:
        return truth, final_preds


def evaluate_f1(truth, preds, class_to_id, dataset_type, coarse=False):
    if ((dataset_type == "multiclass" and 
        all(isinstance(p, str) for p in preds) and 
        all(isinstance(t, str) for t in truth)) or
        (dataset_type == "multilabel" and
        all(isinstance(p, list) for p in preds) and
        all(isinstance(t, list) for t in truth))):
        numerical = False
    else:
        numerical = True

    class_to_id = class_to_id.copy()
    id_to_class = {n: c for c, n in class_to_id.items()}
    if coarse:
        for c, i in class_to_id.items():
            class_to_id[c] = to_coarse(i)

        # Only temporarily defined
        id_to_class_coarse = {i: j for j, i in enumerate(sorted(set(class_to_id.values())))}
        class_to_id_coarse = {c: i for i, c in id_to_class_coarse.items()}

    if dataset_type == "multilabel":
        average = "samples"

        if numerical and coarse:
            # If multihot vectors are provided, and a coarse evaluation is to be done, they need to be re-mapped
            # to new multihot vectors following the amount of coarse ids
            coarse_preds = [[class_to_id[c] for c in multihot_to_list_of_classes(p)] for p in preds]
            coarse_truth = [[class_to_id[c] for c in multihot_to_list_of_classes(p)] for p in truth]

            preds = torch.stack([label_set_to_multihot(p, id_to_class_coarse) for p in coarse_preds]).cpu().numpy()
            truth = torch.stack([label_set_to_multihot(p, id_to_class_coarse) for p in coarse_truth]).cpu().numpy()

        elif not numerical:
            if coarse:
                # The case when preds and truth are lists of narrative ids: get coarse-grained multihot vectors
                preds = torch.stack([label_set_to_multihot(p, id_to_class_coarse, coarse=coarse)
                                    for p in preds]).cpu().numpy()
                truth = torch.stack([label_set_to_multihot(p, id_to_class_coarse, coarse=coarse)
                                    for p in truth]).cpu().numpy()
            else:
                # The case when preds and truth are lists of narrative ids: get fine-grained multihot vectors
                preds = torch.stack([label_set_to_multihot(p, id_to_class, coarse=coarse) for p in preds]).cpu().numpy()
                truth = torch.stack([label_set_to_multihot(p, id_to_class, coarse=coarse) for p in truth]).cpu().numpy()

    else:
        average = "macro"

        if coarse:
            # If evaluating on coarse labels, remap all pred and true classes to the new classes
            # First, get the ids (coarse_preds, coarse_truth)
            if numerical:
                coarse_preds = [class_to_id[c] for c in preds]
                coarse_truth = [class_to_id[c] for c in truth]
            else:
                coarse_preds = list(map(to_coarse, preds))
                coarse_truth = list(map(to_coarse, truth))

            # Then, get back the new classes
            preds = np.array([id_to_class_coarse[p] for p in coarse_preds])
            truth = np.array([id_to_class_coarse[p] for p in coarse_truth])

    if dataset_type == "multilabel":
        labels = (truth.sum(axis=0) > 0).nonzero()[0].tolist()
    else:
        labels = sorted(set(truth))

    if numerical:
        if coarse:
            target_names = [class_to_id_coarse[l] for l in labels]
        else:
            target_names = [class_to_id[l] for l in labels]
    else:
        target_names = None

    return (f1_score(truth, preds, average=average, labels=labels),
            classification_report(truth, preds, labels=labels, target_names=target_names))


def evaluate_ari(truth, cluster_labels, class_to_id, coarse=False):
    if coarse:
        ari = adjusted_rand_score([to_coarse(class_to_id[t]) for t in truth], cluster_labels)
    else:
        ari = adjusted_rand_score(truth, cluster_labels)
    return ari


def evaluate_bcubed(truth, preds, coarse=False, beta=1):
    if not coarse:
        truth_dict = {i: set(t) for i, t in enumerate(truth)}
    else:
        truth_dict = {i: set(map(to_coarse, t)) for i, t in enumerate(truth)}
    preds_dict = {i: set(p) for i, p in enumerate(preds)}
    precision = bcubed.precision(preds_dict, truth_dict)
    recall = bcubed.recall(preds_dict, truth_dict)
    fscore = bcubed.fscore(precision, recall, beta=beta)
    return precision, recall, fscore


def fuzzy_silhouette_score(embs, labels, alpha=1):
    highest_two = np.sort(labels, axis=-1)[:, -2:]
    highest_label = np.argmax(labels, axis=-1)
    silhouettes = silhouette_samples(embs, highest_label, metric="cosine")
    weight = (highest_two[:, 1] - highest_two[:, 0]) ** alpha
    fs = sum(weight * silhouettes) / sum(weight)
    return fs
