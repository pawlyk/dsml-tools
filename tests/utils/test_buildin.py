from dsmlt.utils import (
    dict_of_lists_to_list_of_dicts,
    list_of_dicts_to_dict_of_lists,
    dl_to_ld,
    ld_to_dl,
)

# fmt: off
dict_of_lists_to_list_of_dicts_test_data = {
    'a': ['a', 1, 2, 3, 4, 4, ],
    'b': ['b', 2, 3, 4, 5, 5, ],
    'c': ['c', 3, 4, 5, ],
}

list_of_dicts_to_dict_of_lists_test_data = [
    {'a': 'a', 'b': 'b', 'c': 'c', },
    {'a': 1, 'b': 2, 'c': 3, },
    {'a': 2, 'b': 3, 'c': 4, },
    {'a': 3, 'b': 4, 'c': 5, },
    {'a': 4, 'b': 5, },
    {'a': 4, 'b': 5, },
]
# fmt: on


def test_dict_of_lists_to_list_of_dicts():
    assert (
        dict_of_lists_to_list_of_dicts(
            dict_of_lists_to_list_of_dicts_test_data
        )
        == list_of_dicts_to_dict_of_lists_test_data
    )


def test_list_of_dicts_to_dict_of_lists():
    assert (
        list_of_dicts_to_dict_of_lists(
            list_of_dicts_to_dict_of_lists_test_data
        )
        == dict_of_lists_to_list_of_dicts_test_data
    )


def test_dl_to_ld():
    assert (
        dl_to_ld(dict_of_lists_to_list_of_dicts_test_data)
        == list_of_dicts_to_dict_of_lists_test_data
    )


def test_ld_to_dl():
    assert (
        ld_to_dl(list_of_dicts_to_dict_of_lists_test_data)
        == dict_of_lists_to_list_of_dicts_test_data
    )
