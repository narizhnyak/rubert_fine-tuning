import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

from sklearn.model_selection import train_test_split

import click
import logging
from typing import Union


def get_dataset(
    input_ids: torch.tensor,
    token_type_ids: torch.tensor,
    attention_mask: torch.tensor,
    labels: torch.tensor,
) -> TensorDataset:
    """
    @param input_ids: input_ids from the tokenizer
    @param token_type_ids: token_type_ids from the tokenizer
    @param attention_mask: attention_mask from the tokenizer
    @param labels: object labels
    @return: full TensorDataset object
    """
    return TensorDataset(input_ids, token_type_ids, attention_mask, labels)


def get_train_val_stratified_dataset(
    full_dataset: TensorDataset,
    labels: torch.tensor,
    test_size: float,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    @param full_dataset: full TensorDataset object
    @param labels: object labels
    @param test_size: test sample size
    @return: stratified train and test datasets
    """
    train_indices, val_indices = train_test_split(
        list(range(len(labels))), test_size=test_size, stratify=labels
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    return train_dataset, val_dataset


def get_data_loader(dataset: Union[TensorDataset, torch.utils.data.Subset], batch_size: int) -> DataLoader:
    """
    @param dataset: train or test dataset
    @param batch_size: size of one batch in dataloader object
    @return: dataloader object
    """
    return DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
    )


@click.command()
@click.argument("input_ids_filepath", type=click.Path(exists=True))
@click.argument("token_type_ids_filepath", type=click.Path(exists=True))
@click.argument("attention_mask_filepath", type=click.Path(exists=True))
@click.argument("labels_filepath", type=click.Path(exists=True))
@click.argument("train_dataloader_filepath", type=click.Path())
@click.argument("validation_dataloader_filepath", type=click.Path())
@click.argument("full_dataset_filepath", type=click.Path())
@click.argument("batch_size", type=click.INT)
@click.argument("test_size", type=click.FLOAT)
def main(
    input_ids_filepath: str,
    token_type_ids_filepath: str,
    attention_mask_filepath: str,
    labels_filepath: str,
    train_dataloader_filepath: str,
    validation_dataloader_filepath: str,
    full_dataset_filepath: str,
    batch_size: int,
    test_size: float,
) -> None:
    """
    Save train and validation dataloader objects to /processed
    @param input_ids_filepath: path to an existing input ids file
    @param token_type_ids_filepath: path to an existing token type ids file
    @param attention_mask_filepath: path to an existing attention mask file
    @param labels_filepath: path to an existing labels  file
    @param train_dataloader_filepath: path to the output file train dataloader
    @param validation_dataloader_filepath: path to the output file validation dataloader
    @param full_dataset_filepath: path to the output file full dataset
    @param batch_size: size of one batch in dataloader object
    @param test_size: test sample size
    """
    input_ids = torch.load(input_ids_filepath)
    token_type_ids = torch.load(token_type_ids_filepath)
    attention_mask = torch.load(attention_mask_filepath)
    labels = torch.load(labels_filepath)

    full_dataset = get_dataset(input_ids, token_type_ids, attention_mask, labels)
    train_dataset, val_dataset = get_train_val_stratified_dataset(
        full_dataset, labels, test_size
    )

    train_dataloader = get_data_loader(train_dataset, batch_size)
    validation_dataloader = get_data_loader(val_dataset, batch_size)
    full_dataloader = get_data_loader(full_dataset, batch_size)

    torch.save(train_dataloader, train_dataloader_filepath)
    torch.save(validation_dataloader, validation_dataloader_filepath)
    torch.save(full_dataloader, full_dataset_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
