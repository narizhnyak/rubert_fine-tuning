import elasticsearch.client
from elasticsearch import Elasticsearch

client = Elasticsearch([{"host": "elastic-atlas01.cmo.rubin.local", "port": 9200}])
index_name = "fpg_01.04.2022"
# скрыть!!!


def get_scroll_query() -> elasticsearch.client.Elasticsearch:
    """
    @return: response from ES index
    """
    resp = client.search(
        index=index_name,
        body={"size": 100},
        scroll="3m",  # time value for search
    )
    return resp


def get_scroll_id(resp: elasticsearch.client.Elasticsearch) -> int:
    """
    @param resp: response from ES index
    @return: scroll id
    """
    return resp["_scroll_id"]


def get_doc_count(resp: elasticsearch.client.Elasticsearch) -> int:
    """
    @param resp: response from ES index
    @return: count of docs in index
    """
    return resp["hits"]["total"]["value"]


def scroll_index(scroll_id: int) -> elasticsearch.client.Elasticsearch:
    """
    @param scroll_id: id of current response
    @return: response from ES by the current scroll id
    """
    resp = client.scroll(
        scroll_id=scroll_id,
        scroll="3m",
    )
    return resp


def get_orgs_info(one_org: dict) -> list:
    """
    @param one_org: one organization response
    @return: necessary information from response
    """
    # скрыть!!!
    if one_org is not None:
        if one_org.get("_source").get("organization"):
            org_data = [
                one_org["_source"]["organization"]["ogrn"],
                one_org["_source"]["organization"]["fullname"],
                one_org["_source"]["aimid"],
                one_org["_source"]["statusid"] == 16,
            ]

            return org_data
    else:
        return []
