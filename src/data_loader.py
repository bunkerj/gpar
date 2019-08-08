import os
import numpy as np
import pandas as pd
from src.src_utils import stack_all_columns

DATA_DIR_PATH = os.path.abspath(os.path.join(__file__, '../../data'))


def normalize_data(data_col):
    return (data_col - data_col.mean()) / data_col.std()


def get_column(data, field):
    return data[field].values.reshape((len(data), 1))


def get_index(data):
    return data.index.values.reshape((len(data), 1))


def get_output_data(data):
    return get_column(data, 'index')


def remove_ext(filename):
    return filename.split('.')[0]


def get_ordered_outputs(outputs, labels):
    return [outputs[label] for label in labels]


def get_arbitrary_element(dictionary):
    return list(dictionary.values())[0]


def load_data_from_csv(filename):
    data_file_path = os.path.join(DATA_DIR_PATH, filename)
    return pd.read_csv(data_file_path, thousands=',')


def format_data(inputs, outputs, data_src):
    labels = outputs.keys() if data_src is None else data_src
    ordered_outputs = get_ordered_outputs(outputs, labels)
    return get_arbitrary_element(inputs), stack_all_columns(ordered_outputs), labels


def are_identical(dictionary):
    values = list(dictionary.values())
    if len(values) == 0:
        return True

    reference = values[0]
    for element in values[1:]:
        if not np.array_equal(reference, element):
            return False

    return True


def get_filenames(data_src):
    if data_src is None:
        return os.listdir(DATA_DIR_PATH)
    else:
        custom_list = []
        for filename in os.listdir(DATA_DIR_PATH):
            if remove_ext(filename) in data_src:
                custom_list.append(filename)
        if len(data_src) != len(custom_list):
            raise Exception('Could not find all relevant data files')
        return custom_list


def get_processed_data(data_src=None):
    inputs = {}
    outputs = {}

    for filename in get_filenames(data_src):
        filename_no_ext = remove_ext(filename)
        data = load_data_from_csv(filename)
        inputs[filename_no_ext] = get_index(data)
        outputs[filename_no_ext] = get_column(data, 'close')

    if not are_identical(inputs):
        raise Exception('Inputs are not identical')

    return format_data(inputs, outputs, data_src)
