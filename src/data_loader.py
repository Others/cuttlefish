import random
from glob import glob

import torch
from torch.utils.data import IterableDataset

from print_with_timestamp import print_with_timestamp


class MoveVectorLoader(IterableDataset):
    def __init__(self, name, files, cap=None):
        self.files = files
        self.cap = int(cap) if cap is not None else None
        print_with_timestamp(
            f"{name} Loader with {len(self.files)} files, and a cap of {self.cap}"
        )

    def shuffle(self):
        random.shuffle(self.files)

    def __iter__(self):
        copied_list = list(self.files)
        cap = self.cap

        def generator():
            n = 0
            for file in copied_list:
                try:
                    data_l = torch.load(file)
                    for idx in range(len(data_l["data"])):
                        n += 1
                        if cap is not None and n > cap:
                            return
                        yield data_l["data"][idx], data_l["labels"][idx]
                except Exception as e:
                    print(f"File read error, failing open {e}")

        return iter(generator())


def create_lazy_loaders(
    path, cap=None, test_fraction=0.2, validation_fraction=(0.2 * 0.8)
):
    file_paths = glob(f"{path}/*_pgns_*.pt")
    random.shuffle(file_paths)

    n_paths = len(file_paths)

    print_with_timestamp(f"Found {n_paths} data files")

    test_set = []
    while len(test_set) < n_paths * test_fraction:
        test_set.append(file_paths.pop())

    validation_set = []
    while len(validation_set) < n_paths * validation_fraction:
        validation_set.append(file_paths.pop())

    training_set = file_paths
    if cap is None:
        return (
            MoveVectorLoader("Training", training_set),
            MoveVectorLoader("Validation", validation_set),
            MoveVectorLoader("Test", test_set),
        )
    else:
        return (
            MoveVectorLoader(
                "Training",
                training_set,
                cap * (1 - test_fraction - validation_fraction),
            ),
            MoveVectorLoader("Validation", validation_set, cap * validation_fraction),
            MoveVectorLoader("Test", test_set, cap * test_fraction),
        )


def custom_collate(batch):
    # TODO: This should probably stack the vectors, rather than making the training step do that
    data = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return {"data": data, "labels": labels}
