import torch
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import click
import logging
from os import remove
from os.path import exists

from src.models.train_model import get_device


def get_all_metrics(preds: np.ndarray, true_label_ids: np.ndarray) -> dict:
    """
    Returns dictionary of all necessary metrics
    @param preds: flatted model predictions
    @param true_label_ids: flatted true labels
    @return: metric values
    """

    return {
        "accuracy_score": accuracy_score(preds, true_label_ids),
        "recall_score": recall_score(preds, true_label_ids, average="micro"),
        "precision_score": precision_score(preds, true_label_ids, average="micro"),
        "f1_score": f1_score(preds, true_label_ids, average="micro"),
    }


def get_preds_and_labels_flatted(
    logits: np.ndarray, true_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    @param logits: model predictions
    @param true_ids: true labels
    @return: flatted model predictions and true labels
    """
    preds_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = true_ids.flatten()

    return preds_flat, labels_flat


def file_put_contents(filename, data) -> None:
    """
    @param filename: file to append data
    @param data: appended data
    @return:
    """
    with open(filename, 'a', encoding='utf-8') as f_out:
        f_out.write(data + "\n")
    return


def val_model(
    model: any,
    validation_dataloader: DataLoader,
    device: str,
    model_path: str
) -> None:

    log_filename = 'validation.log'
    model_full_path = model_path + log_filename

    model.eval()

    eval_metrics = {
        "accuracy_score": 0,
        "recall_score": 0,
        "precision_score": 0,
        "f1_score": 0,
    }

    eval_loss = 0

    for batch in validation_dataloader:

        b_input_ids = torch.tensor(batch[0]).to(device).long()
        b_token_type_ids = torch.tensor(batch[1]).to(device).long()
        b_attention_mask = torch.tensor(batch[2]).to(device).long()
        b_labels = torch.tensor(batch[3]).to(device).long()

        with torch.no_grad():

            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )

        loss = outputs.loss
        logits = outputs.logits

        eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        flat_preds, flat_label_ids = get_preds_and_labels_flatted(logits, label_ids)
        metric_results = get_all_metrics(flat_preds, flat_label_ids)

        for metric in metric_results.keys():
            eval_metrics[metric] += metric_results[metric]

    for metric in eval_metrics.keys():
        metric_value = eval_metrics[metric] / len(validation_dataloader)
        if exists(model_full_path):
            remove(model_full_path)
        file_put_contents(model_full_path, f'{metric}: {metric_value:.4f}')

    return


@click.command()
@click.argument("validation_dataloader_filepath", type=click.Path(exists=True))
@click.argument("partial_trained_model_filepath", type=click.Path(exists=True))
def main(
    validation_dataloader_filepath: str,
    partial_trained_model_filepath: str
) -> None:
    """
    Writes model metrics in a log file
    @param validation_dataloader_filepath: path to validation dataloader file
    @param partial_trained_model_filepath: path to partial trained model file
    """
    path_to_main_folder = partial_trained_model_filepath.replace('pytorch_model.bin', '')
    device = get_device()
    model = torch.load(partial_trained_model_filepath)
    validation_dataloader = torch.load(validation_dataloader_filepath)

    val_model(model, validation_dataloader, device, path_to_main_folder)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
