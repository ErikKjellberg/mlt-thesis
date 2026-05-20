from typing import Union
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from utils import multihot_to_list_of_classes


def get_flat_cmap(classes):
    n_classes = len(classes)
    hues = np.linspace(0, 1, n_classes, endpoint=False)
    saturations = np.full(n_classes, 1)
    values = np.tile(np.linspace(0.5, 1, 3), n_classes // 3 + 1)[:n_classes]
    hsv = np.stack([hues, saturations, values], axis=1)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return mpl.colors.ListedColormap(rgb)


def get_hierarchical_cmap(id_to_class):
    """
        Assigns a color to each id in the keys of 'id_to_class'.
        Id:s belonging to the same coarse category get the same hue, but different values (lightness).
    """
    classes = set(id_to_class.keys())
    n_classes = len(classes)
    hierarchical_classes = dict()
    for l in sorted(classes.difference({"0"})):
        if l != "0":
            s, sn = map(int, l.split("_"))
            if s not in hierarchical_classes:
                hierarchical_classes[s] = []
            hierarchical_classes[s].append(sn)

    hues = np.full(n_classes, 0.5)
    saturations = np.full(n_classes, 0.5)
    values = np.full(n_classes, 0.5)

    n_hues_except_grey = len(hierarchical_classes.keys())
    hue_per_class = np.linspace(0, 1, n_hues_except_grey, endpoint=False)
    l_to_c = {n: dict() for n in hierarchical_classes}

    for i, n in enumerate(hierarchical_classes.keys()):
        value_per_subclass = np.linspace(0.5, 1, len(hierarchical_classes[n]))
        for j, sn in enumerate(hierarchical_classes[n]):
            hues[id_to_class[str(n)+"_"+str(sn)]] = hue_per_class[i]
            values[id_to_class[str(n)+"_"+str(sn)]] = value_per_subclass[j]

    if "0" in id_to_class:
        # "Other" gets grey as its color
        hues[id_to_class["0"]] = 0
        saturations[id_to_class["0"]] = 0
        values[id_to_class["0"]] = 0.7
    hsv = np.stack([hues, saturations, values], axis=1)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return mpl.colors.ListedColormap(rgb)


def plot_examples(classes, embs, ax, class_to_id, include_labels, cmap, ncols, id_to_narrative=None, loc=None):
    # Helper function to plot `embs` colored by `classes` on the axis `ax`.
    unique_clusters = set()
    for c in classes:
        unique_clusters.add(c)
    n_classes = len(unique_clusters)
    ax.set_xticks([])
    ax.set_yticks([])
    for label in sorted(unique_clusters):
        cluster_embs = embs[classes == label]
        x = cluster_embs[:, 0]
        y = cluster_embs[:, 1]
        if include_labels:
            label_id = class_to_id[int(label)]
            if id_to_narrative:
                legend_label = f"{label_id}: {id_to_narrative[label_id]}"
            else:
                legend_label = f"{label_id}"
            ax.scatter(x, y, color=cmap(label), s=75/ncols, label=legend_label, edgecolors='black')
        else:
            ax.scatter(x, y, color=cmap(label), s=75/ncols, edgecolors='black')
    if include_labels:
        if loc:
            ax.legend(title="Clusters", loc=loc)
        else:
            ax.legend(title="Clusters")

def plot_examples_multilabel(classes, embs, class_to_id, cmap, ncols=3, id_to_narrative=None, loc=None):
    # Use one plot for each class
    unique_clusters = set()
    for cs in classes:
        for c in cs:
            unique_clusters.add(c)
    n_classes = len(unique_clusters)

    nrows = math.ceil(n_classes/ncols)

    plot_w = 20
    plot_h = plot_w * math.ceil(n_classes/ncols) / ncols

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    fig.set_figwidth(plot_w)
    fig.set_figheight(plot_h)
    ax = axs

    for t, label in enumerate(sorted(unique_clusters)):
        if n_classes <= ncols:
            ax = axs[t]
        else:
            ax = axs[t // ncols][t % ncols]

        x = embs[:, 0]
        y = embs[:, 1]
        ax.scatter(x, y, color=(0.8, 0.8, 0.8), s=75/ncols)

        cluster_embs = embs[np.where([label in c for c in classes])]
        x = cluster_embs[:, 0]
        y = cluster_embs[:, 1]
        label_id = class_to_id[int(label)]
        if id_to_narrative:
            legend_label = f"{label_id}: {id_to_narrative[label_id]}"
        else:
            legend_label = f"{label_id}"
        ax.scatter(x, y, color=cmap(label), s=75/ncols, label=legend_label, edgecolors='black')
        if loc:
            ax.legend(title="Clusters", loc=loc)
        else:
            ax.legend(title="Clusters")


def plot_results(
        truth : np.ndarray, 
        preds : Union[np.ndarray, list[set]], 
        embs : np.ndarray, 
        id_to_class, 
        dataset_type, 
        id_to_narrative=None, 
        use_same_cmap=False, 
        loc=None, 
        save_path=None):
    """
        Plots `embs` colored by the values `preds` and `truth`, in separate plots.
        Works for both multiclass and multi-label datasets.
    """
    class_to_id = {c: i for i, c in id_to_class.items()}
    cmap_true = get_hierarchical_cmap(id_to_class)

    if dataset_type == "multiclass":
        pred_label_mapping = {c: i for i, c in enumerate(np.unique(preds))}
        rev_mapping = {i: c for c, i in pred_label_mapping.items()}
        cmap_pred = get_flat_cmap(pred_label_mapping.values())
        mapped_preds = np.array([pred_label_mapping[c] for c in preds])
        
        ncols = 2
        fig, axs = plt.subplots(ncols=ncols, nrows=1)
        fig.set_figwidth(20)
        fig.set_figheight(20 / ncols)

        plot_examples(truth, embs, axs[0], class_to_id, True, cmap_true, ncols, id_to_narrative=id_to_narrative, loc=loc)
        if use_same_cmap:
            plot_examples(preds, embs, axs[1], class_to_id, True, cmap_true, ncols, id_to_narrative=id_to_narrative, loc=loc)
        else:
            plot_examples(mapped_preds, embs, axs[1], rev_mapping, True, cmap_pred, ncols, loc=loc)
        axs[0].set_title("True")
        axs[1].set_title("Predicted")

    else:
        # Plot the true labels
        classes = list(map(multihot_to_list_of_classes, truth))
        plot_examples_multilabel(classes, embs, class_to_id, cmap_true)

        # Plot the true labels
        if isinstance(preds, np.ndarray) and len(preds.shape) == 2:
            preds = [set(multihot_to_list_of_classes(p)) for p in preds]

        if all(isinstance(p, set) for p in preds):
            unique_labels = set().union(*preds)
        elif isinstance(preds, np.ndarray) and len(preds.shape == 1):
            unique_labels = np.unique(preds)
            
        pred_label_mapping = {c: i for i, c in enumerate(unique_labels)}
        rev_mapping = {i: c for c, i in pred_label_mapping.items()}
        cmap_pred = get_flat_cmap(pred_label_mapping.values())

        if all(isinstance(p, set) for p in preds):
            plot_examples_multilabel(preds, embs, rev_mapping, cmap_pred)
            
        elif isinstance(preds, np.ndarray) and len(preds.shape == 1):
            fig, axs1 = plt.subplots(ncols=1, nrows=1)
            plot_examples(preds, embs, axs1, rev_mapping, True, cmap_pred, 1)
            axs1.set_title("Predicted")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_hierarchical_results(
        truth, 
        coarse_preds, 
        fine_preds, 
        embs, 
        id_to_class, 
        id_to_narrative=None, 
        loc=None, 
        save_path=None):
    """
        Plots `embs` colored by `coarse_preds` and `fine_preds`, the coarse and fine predictions respectively, along 
        with the true labels `truth`.
    """
    class_to_id = {c: i for i, c in id_to_class.items()}
    cmap_true = get_hierarchical_cmap(id_to_class)
    
    pred_label_mapping = {c:i for i,c in enumerate(np.unique(coarse_preds))}
    rev_mapping = {i:c for c,i in pred_label_mapping.items()}
    cmap_pred = get_flat_cmap(pred_label_mapping.values())
    mapped_preds = np.array([pred_label_mapping[c] for c in coarse_preds])

    fine_pred_label_mapping = {c:i for i,c in enumerate(np.unique(fine_preds))}
    fine_rev_mapping = {i:c for c,i in fine_pred_label_mapping.items()}
    fine_cmap_pred = get_flat_cmap(fine_pred_label_mapping.values())
    fine_mapped_preds = np.array([fine_pred_label_mapping[c] for c in fine_preds])

    ncols=3
    fig, axs = plt.subplots(ncols=ncols, nrows=1)
    fig.set_figwidth(25)
    fig.set_figheight(25 / ncols)

    plot_examples(truth, embs, axs[0], class_to_id, True, cmap_true, ncols, id_to_narrative=id_to_narrative, loc=loc)
    plot_examples(mapped_preds, embs, axs[1], rev_mapping, True, cmap_pred, ncols, loc=loc)
    plot_examples(fine_mapped_preds, embs, axs[2], fine_rev_mapping, True, fine_cmap_pred, ncols, loc=loc)

    axs[0].set_title("True")
    axs[1].set_title("Predicted (coarse)")
    axs[2].set_title("Predicted (fine)")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
