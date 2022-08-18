# -*- coding: utf-8 -*-
import csv
import click
import logging
from math import ceil
from src.data.make_ds_helper import (
    get_scroll_query,
    get_orgs_info,
    get_doc_count,
    get_scroll_id,
    scroll_index,
)


def append_to_csv(filename: str, row: list) -> None:
    """
    append a data row to the csv-file
    @param filename: csv-file for appending data
    @param row: data row
    """
    with open(filename, "a", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(output_filepath: str) -> None:
    """
    scroll through the whole ES index and save the data from it
    @param output_filepath: path to raw dataset file
    """

    append_to_csv(output_filepath, ["text", "add1", "add2", "target"])

    response = get_scroll_query()
    for one_org in response["hits"]["hits"]:
        org_data = get_orgs_info(one_org)
        if org_data:
            append_to_csv(output_filepath, org_data)

    doc_count = get_doc_count(response)
    scroll_id = get_scroll_id(response)
    scroll_count = ceil(doc_count / 100)

    cur_scroll_count = 0

    while cur_scroll_count != scroll_count and cur_scroll_count < 20:
        orgs_data = scroll_index(scroll_id)["hits"]["hits"]
        cur_scroll_count = cur_scroll_count + 1
        for one_org in orgs_data:
            org_data = get_orgs_info(one_org)
            if org_data:
                append_to_csv(output_filepath, org_data)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
