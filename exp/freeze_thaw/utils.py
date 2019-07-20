def hyp_to_key(hyp):
    return '_'.join(str(v) for v in hyp)


def key_to_hyp(key):
    return tuple(int(v) for v in key.split('_'))
