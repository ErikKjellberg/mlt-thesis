import torch
import numpy as np

# Global values
LETTER_TO_NUM = {l: i+1 for i, l in enumerate("abcdefghijklmnopqrstuvwxyz")}


def avg(l):
    return sum(l)/len(l)


def to_coarse(l):
    return l.split("_")[0]


def label_set_to_multihot(ex, narrative_to_class, coarse=False):
    return torch.zeros(
        len(narrative_to_class.keys())).index_fill_(
        0, torch.tensor([narrative_to_class[to_coarse(n) if coarse else n] for n in set(ex)],
                        dtype=torch.int64),
        1)


def logits_to_multihot(logits, min_p=0.5):
    # Keep only classes each having a score of at least min_p.
    # Forces argmax prediction if no class gets min_p or higher.
    scores = logits.sigmoid()
    pred = torch.zeros_like(scores)
    above_threshold = (scores > min_p)
    no_pred = (~above_threshold).all(dim=-1)
    pred[above_threshold] = 1
    pred[no_pred, scores.argmax(dim=-1)[no_pred]] = 1
    return pred.int()


def multihot_to_list_of_classes(multihot):
    return np.where(multihot == 1)[0]


def get_taxonomy_string(taxonomy_dict, max_depth=1):
    taxonomy_list = []

    def iter_taxonomy(tax, depth, pre):
        for k, v in tax.items():
            if k != "description":
                if depth == 0:
                    taxonomy_list.append("Narrative " + pre + k + ". " + v["description"])
                else:
                    taxonomy_list.append(pre + k + ". " + v["description"])
                if depth < max_depth:
                    taxonomy_list.append("Subnarratives:")
                    iter_taxonomy(v, depth + 1, pre + k + ".")

    iter_taxonomy(taxonomy_dict, 0, "")
    taxonomy = "\n".join(taxonomy_list)
    return taxonomy


def get_id_to_narrative_dict(taxonomy, max_depth, dataset_name, convert_to_num=False):
    id_to_narrative = dict()

    def get_id_to_narrative_rec(tax, depth, pre_id, pre_des):
        for k, v in tax.items():
            if k != "description":
                if convert_to_num:
                    try:
                        k = str(LETTER_TO_NUM[k])
                    except:
                        pass
                if depth == 0 or dataset_name == "polynarrative":
                    id_to_narrative[pre_id + "0"] = pre_des + "Other"
                if depth > 0:
                    id_to_narrative[pre_id + k] = pre_des + v["description"].split(":")[0]
                if depth < max_depth:
                    get_id_to_narrative_rec(v, depth + 1, pre_id + k + "_",
                                            pre_des + v["description"].split(":")[0] + ": ")

    get_id_to_narrative_rec(taxonomy, 0, "", "")
    return id_to_narrative


def process_response(r, convert_num=False):
    # For chat models
    if r == "Other":
        return "0"
    try:
        n, s = r.split(".")
        s = LETTER_TO_NUM[s]
        return n+"_"+str(s)
    except:
        return "0"


def process_response_multilabel(rs, convert_num=False):
    # For multi-label chat models
    ps = []
    if isinstance(rs, str):
        rs = rs.split("; ")
    for r in rs:
        ps.append(process_response(r, convert_num))
    return list(set(ps))
