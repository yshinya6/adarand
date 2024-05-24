import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import train_supervised_classifier as train
import util.yaml_utils as yaml_utils
from util.train_util import load_models


def calc_top5_acc(pred, t):
    top5_preds = pred.argsort()[:, -5:]
    return np.asarray(np.any(top5_preds.T == t, axis=0).mean(dtype="f"))


def main():
    schema = "filename\tTop1\tTop5\tF-Score\tPrecision\tRecall\tF-score"
    parser = argparse.ArgumentParser(description=f"Target Model Tester \n ({schema})")
    parser.add_argument("--config_path", type=str, default="configs/base.yml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="./result/", help="directory to save the results to")
    parser.add_argument("--process_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    pattern = "-".join([config.pattern, config.models["classifier"]["name"], config.dataset["dataset_name"]])
    out_path = os.path.join(args.results_dir, pattern, f"expr{args.process_num}")
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    # Model
    with open(os.path.join(out_path, "log")) as f:
        log = json.load(f)
    model_path = log["best_model"]
    classifier = load_models(config.models["classifier"])
    classifier = torch.nn.DataParallel(classifier, device_ids=[0])
    classifier.load_state_dict(torch.load(model_path))
    classifier.to(device)

    # Dataset
    test_dataset = yaml_utils.load_dataset(config["dataset"], test=True)
    test_loader = DataLoader(test_dataset, config.batchsize, shuffle=False, num_workers=16)

    # Test loop
    pred_labels = []
    correct_labels = []
    count = 0
    classifier.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            result = classifier(data)
            pred_labels.append(result.cpu().data.numpy())
            correct_labels.append(np.array(labels.cpu().data.numpy()))
            count += 1

    # Evaluation
    pred_labels = np.concatenate(pred_labels)
    correct_labels = np.concatenate(correct_labels)
    pred_label_top1 = pred_labels.argsort()[:, -1]
    top1 = accuracy_score(correct_labels, pred_label_top1)
    top5 = calc_top5_acc(pred_labels, correct_labels)
    precision = precision_score(correct_labels, pred_label_top1, average="macro")
    recall = recall_score(correct_labels, pred_label_top1, average="macro")
    f_score = f1_score(correct_labels, pred_label_top1, average="macro")
    out_results = {
        "accuracy": float(top1),
        "top-5 accuracy": float(top5),
        "precision": float(precision),
        "recall": float(recall),
        "f-score": float(f_score),
    }

    # Report
    result_path = os.path.join(out_path, "test_result.yaml")
    if os.path.exists(result_path):
        result_yaml = yaml.load(open(result_path, "r+"), Loader=yaml.SafeLoader)
    else:
        result_yaml = {}
    result_yaml.update(out_results)
    with open(result_path, mode="w") as f:
        f.write(yaml.dump(result_yaml, default_flow_style=False))

    print(f"{pattern}\t{top1}\t{top5}\t{precision}\t{recall}\t{f_score}")
    return out_results


if __name__ == "__main__":
    main()
