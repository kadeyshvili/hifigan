import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    result_batch['wav'] = torch.stack([item['wav'] for item in dataset_items])
    result_batch['melspec'] = torch.stack([item['melspec'] for item in dataset_items])
    result_batch['path'] = [item['path'] for item in dataset_items]
    return result_batch