from collections import defaultdict
from typing import Dict, List, Union


__all__ = (
    "dict_of_lists_to_list_of_dicts",
    "list_of_dicts_to_dict_of_lists",
    "dl_to_ld",
    "ld_to_dl",
)


def dict_of_lists_to_list_of_dicts(
    data: Dict[str, List[Union[str, int, float]]]
) -> List[Dict]:
    result = [{} for _ in range(max(map(len, data.values())))]
    for key, seq in data.items():
        for d, value in zip(result, seq):
            d[key] = value

    return result


def list_of_dicts_to_dict_of_lists(
    data: List[Dict],
) -> Dict[str, List[Union[str, int, float]]]:
    out_data = defaultdict(list)
    for d in data:
        for key, val in d.items():
            out_data[key].append(val)

    return out_data


dl_to_ld = dict_of_lists_to_list_of_dicts
ld_to_dl = list_of_dicts_to_dict_of_lists
