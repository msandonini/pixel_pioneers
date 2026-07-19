import random


def split_by_ref(ref_paths, val_frac: float = 0.16, test_frac: float = 0.16):
    unique_refs = sorted(set(ref_paths), key=str)
    random.shuffle(unique_refs)

    n = len(unique_refs)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_refs = set(unique_refs[:n_test])
    val_refs = set(unique_refs[n_test:n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for i, ref in enumerate(ref_paths):
        if ref in test_refs:
            test_idx.append(i)
        elif ref in val_refs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx, test_idx