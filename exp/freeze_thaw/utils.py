import tensorflow as tf


def hyp_to_key(hyp):
    return '_'.join(str(v) for v in hyp)


def key_to_hyp(key):
    return tuple(int(v) for v in key.split('_'))


def concat_tensors(tensor_list, axis):
    filtered_tensor_list = list(filter(lambda x: x is not None, tensor_list))
    return tf.concat(filtered_tensor_list, axis)


def get_block_diag_matrix(matrix_list):
    col_index = 0
    current_matrix = None
    total_columns = sum(int(tf.shape(mat)[1]) for mat in matrix_list)
    for mat in matrix_list:
        n_rows, n_cols = tf.shape(mat)
        front_padding = None if int(col_index) == 0 else tf.zeros((n_rows, col_index))
        back_padding = None if int(col_index + n_cols) == total_columns else tf.zeros(
            (n_rows, total_columns - (col_index + n_cols)))
        sub_matrix = concat_tensors([front_padding, mat, back_padding], 1)
        col_index += n_cols
        if current_matrix is None:
            current_matrix = sub_matrix
        else:
            current_matrix = tf.concat([current_matrix, sub_matrix], axis=0)
    return current_matrix
