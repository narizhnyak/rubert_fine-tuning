import torch

from transformers import AutoTokenizer

import numpy as np
import pandas as pd

import logging
import click


def get_tokenizer(model_checkpoint: str) -> any:
    """
    @param model_checkpoint: name of the model checkpoint
    @return: tokenizer object
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def get_input_ids_max_len(df_text_column: pd.Series, tokenizer: any) -> int:
    """
    @param df_text_column: column to be processed by the tokenizer
    @param tokenizer: tokenizer object
    @return: max length of input ids from the tokenizer
    """
    input_ids = df_text_column.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    return max(input_ids.apply(len))


def tokenize_text_field(df_text_column: pd.Series, tokenizer: any, max_len: int) -> pd.Series:
    """
    @param df_text_column: column to be processed by the tokenizer
    @param tokenizer: tokenizer object
    @param max_len: max length of input ids from the tokenizer
    @return: pandas column with dictionary of input_ids, token_type_ids and attention_mask
    for each text in the dataframe
    """
    return df_text_column.apply(
        lambda x: tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,  # padding='longest' does not work correctly in this version
            return_attention_mask=True,
            truncation=True,
        )
    )


def get_offset_and_specify_torch_datatype(
    tokenized_text: pd.Series, offset: str
) -> torch.Tensor:
    """
    @param tokenized_text: pandas column with dictionary of input_ids, token_type_ids and attention_mask
    for each text in the dataframe
    @param offset: offset from the tokenized text dictionary
    @return: torch tensor of specified data
    """
    return torch.tensor(tokenized_text.apply(lambda x: x[offset]), dtype=torch.float64)


@click.command()
@click.argument("df_filepath", type=click.Path(exists=True))
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("input_ids_filepath", type=click.Path())
@click.argument("token_type_ids_filepath", type=click.Path())
@click.argument("attention_mask_filepath", type=click.Path())
@click.argument("labels_filepath", type=click.Path())
def main(
    df_filepath: str,
    model_checkpoint: str,
    input_ids_filepath: str,
    token_type_ids_filepath: str,
    attention_mask_filepath: str,
    labels_filepath: str,
) -> None:
    """
    Save interim data (input ids, token type ids, attention mask, object labels)
    @param df_filepath: path to the raw dataframe file
    @param model_checkpoint: name of the model checkpoint
    @param input_ids_filepath: path to the interim data file with input ids
    @param token_type_ids_filepath: path to the interim data file with token type ids
    @param attention_mask_filepath: path to the interim data file with attention mask
    @param labels_filepath: path to the interim data file with object labels
    """

    df = pd.read_csv(df_filepath)
    df.target = df.target - np.repeat(1, df.shape[0])
    tokenizer = get_tokenizer(model_checkpoint)
    text_column = df.text

    max_len = get_input_ids_max_len(text_column, tokenizer)
    tokenized_text = tokenize_text_field(text_column, tokenizer, max_len)

    input_ids = get_offset_and_specify_torch_datatype(
        tokenized_text, "input_ids"
    )
    torch.save(input_ids, input_ids_filepath)

    token_type_ids = get_offset_and_specify_torch_datatype(
        tokenized_text, "token_type_ids"
    )
    torch.save(token_type_ids, token_type_ids_filepath)

    attention_mask = get_offset_and_specify_torch_datatype(
        tokenized_text, "attention_mask"
    )
    torch.save(attention_mask, attention_mask_filepath)

    labels = torch.tensor(df.target, dtype=torch.float64)
    torch.save(labels, labels_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
