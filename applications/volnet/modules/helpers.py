def parse_range_string(range_string: str):
    return [i for i in range(*map(int, range_string.split(':')))]


def parse_slice_string(slice_string: str):
    def int_or_none(x):
        try:
            out = int(x)
        except ValueError:
            out = None
        return out
    out = tuple(map(int_or_none, slice_string.split(':')))
    return out


def is_integer_index(idx):
    try:
        int_idx = int(idx)
    except Exception():
        return False
    else:
        return int_idx == idx


def is_valid_index_position(pos, idx):
    l = len(idx)
    if pos < -l:
        return False
    if pos >= l:
        return False
    return True

