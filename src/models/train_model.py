import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification

from typing import Union

import time
import click
import logging

from os import path
from os import listdir


def get_device() -> str:
    """
    @return: device name
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_model(model_checkpoint: str, num_labels: int) -> any:
    """
    @param model_checkpoint: name of the model checkpoint
    @param num_labels: number of labels in the dataframe
    @return: model object
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
    )


def get_data_loader(
    dataset: Union[TensorDataset, torch.utils.data.Subset], batch_size: int
) -> DataLoader:
    """
    @param dataset: train or test dataset
    @param batch_size: size of one batch in dataloader object
    @return: dataloader object
    """
    return DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
    )


def train_model(
    model: any,
    device: str,
    num_epochs: int,
    train_dataloader: DataLoader,
) -> any:
    """
    @param model: model object
    @param device: device name
    @param num_epochs: number of model learning epochs
    @param train_dataloader: processed dataloader object for model training
    @return: trained model object
    """

    batch_size = train_dataloader.batch_size
    total_steps = len(train_dataloader) * num_epochs

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.to(device)

    for epoch_i in range(num_epochs):

        t0 = time.time()
        train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = torch.tensor(batch[0]).to(device).long()
            b_token_type_ids = torch.tensor(batch[1]).to(device).long()
            b_attention_mask = torch.tensor(batch[2]).to(device).long()
            b_labels = torch.tensor(batch[3]).to(device).long()

            model.zero_grad()

            outputs = model(
                input_ids=b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )

            loss = outputs.loss

            train_loss = loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % 50 == 0 and not step == 0:
                spent = time.time() - t0

                current_loss = train_loss / batch_size

                print(
                    "Batch {:}  of  {:}.    Spent: {:}. Current_loss {:}".format(
                        step, len(train_dataloader), spent, current_loss
                    )
                )

        avg_train_loss = train_loss / len(train_dataloader)
        training_time = time.time() - t0

        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(training_time))

    return model


def get_path_and_model_index(default_path: str, new_path: bool) -> tuple[str, int]:
    """
    @param default_path: name of model directory
    @param new_path: flag that necessary get path for new model
    @return: actual path and index for new model
    """
    digits = ""
    if path.exists(default_path):
        for i in range(len(default_path)):
            current_symbol = default_path[-i - 1]
            if current_symbol.isdigit():
                digits = current_symbol + digits
            elif digits:
                new_index = int(digits)
                if new_path:
                    new_index += 1
                return default_path[:-i] + str(new_index), new_index

    return default_path, 0


def get_actual_path(files_path: str, new_path: bool) -> str:
    """
    @param files_path: path to save model
    @param new_path: flag that necessary get path for new model
    @return: actual directory for new fine-tuned model
    """
    names = listdir(files_path)
    actual_path, index = files_path + "model0", 0

    for name in names:
        current_path, current_index = get_path_and_model_index(files_path + name, new_path)
        if current_index > index:
            index = current_index
            actual_path = current_path

    return actual_path


@click.command()
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("train_dataloader_filepath", type=click.Path(exists=True))
@click.argument("num_epochs", type=click.INT)
@click.argument("full_train", type=click.BOOL)
def main(
    model_checkpoint: str,
    train_dataloader_filepath: str,
    num_epochs: int = 3,
    full_train: bool = False
) -> None:
    """
    Saving trained model to file
    @param model_checkpoint: model object
    @param train_dataloader_filepath: path to training dataloader file
    @param num_epochs: number of model learning epochs
    @param full_train: flag that train will be on full dataset
    """
    device = get_device()
    model = get_model(model_checkpoint, 13)
    train_dataloader = torch.load(train_dataloader_filepath)

    if full_train:
        path_to_main_folder = "models/full_trained_models/"
    else:
        path_to_main_folder = "models/partial_trained_models/"

    train_model(model, device, num_epochs, train_dataloader)

    actual_path = get_actual_path(path_to_main_folder, True)
    model.save_pretrained(actual_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
