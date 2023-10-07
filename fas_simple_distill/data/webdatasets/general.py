import torch
import webdataset as wds

from . import custom_webdataset as cwds

from braceexpand import braceexpand


def dict_to_tensor_transform(target_dict):
    return torch.tensor(
        [v for v in target_dict.values()], dtype=torch.long
    )  # pylint: disable=not-callable


def single_label_transform(target_dict):
    return float(target_dict["label"])


def get_loader(
    urls,
    dataset_size,
    batch_size,
    transform,
    target_transform=single_label_transform,
    num_workers=4,
    partial_batch=True,
    ddp_equalize=False,
    epoch_shuffle=False,
    shuffle=True,
    split_by_node=True,
    split_by_worker=True,
    **kwargs,
) -> wds.Processor:
    dataset = cwds.WebDataset(
        urls,
        epoch_shuffle=epoch_shuffle,
        shuffle=shuffle,
        split_by_node=split_by_node,
        split_by_worker=split_by_worker,
        **kwargs,
    )

    dataset = (
        dataset.decode("pil")
        .to_tuple("png pickle")
        .map_tuple(transform, target_transform)
        .batched(batch_size, partial=partial_batch)
    )

    loader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers,
    )

    loader.length = dataset_size // batch_size
    if ddp_equalize:
        loader: wds.Processor = loader.ddp_equalize(dataset_size // batch_size)

    return loader

def get_multi_loader(
    urls,
    dataset_size,
    batch_size,
    transform,
    target_transform=single_label_transform,
    num_workers=4,
    partial_batch=True,
    ddp_equalize=False,
    epoch_shuffle=False,
    shuffle=True,
    split_by_node=True,
    split_by_worker=True,
    **kwargs,
) -> wds.Processor:

    exp_urls = []
    for u in urls:
        exp_urls.extend(list(braceexpand(u)))
    urls = exp_urls
    dataset = cwds.WebDataset(
        urls,
        epoch_shuffle=epoch_shuffle,
        shuffle=shuffle,
        split_by_node=split_by_node,
        split_by_worker=split_by_worker,
        **kwargs,
    )

    dataset = (
        dataset.decode("pil")
        .to_tuple("png pickle")
        .map_tuple(transform, target_transform)
        .batched(batch_size, partial=partial_batch)
    )

    loader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers,
    )

    loader.length = dataset_size // batch_size
    if ddp_equalize:
        loader: wds.Processor = loader.ddp_equalize(dataset_size // batch_size)

    return loader
