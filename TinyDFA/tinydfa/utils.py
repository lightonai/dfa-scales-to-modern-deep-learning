def remove_indices(array, indices):
    # From: https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    return [e for i, e in enumerate(array) if i not in set(indices)]

def keep_indices(array, indices):
    return [e for i, e in enumerate(array) if i in set(indices)]
