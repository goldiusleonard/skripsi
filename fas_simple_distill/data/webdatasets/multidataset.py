from typing import Any, Dict, List, Callable, Optional, Union

import torch
from braceexpand import braceexpand
import webdataset as wds
from PIL import Image

import numpy as np

import torch
import torchvision.transforms as tt
import torchvision as t

# import sys
# sys.path.append("../..")
from os.path import join
from os import listdir

from torch.utils.data import Dataset

# import sys
# sys.path.append("wb_augment/WB_color_augmenter/WBAugmenter_Python/")
from .WBAugmenter import WBEmulator as wbAug
from . import WBAugmenter

# from .wb_augment.WB_color_augmenter.WBAugmenter_Python.WBAugmenter import WBEmulator as WbAug
from random import random

TFM = Callable[[Union[torch.Tensor, Image.Image]], torch.Tensor]
TTFM = Callable[[Dict[str, Any]], torch.Tensor]


class MultiUnlimitedDataLoader:
    def __init__(
        self,
        urls: List[Union[str, List[str]]],
        num_workers: int,
        batch_size: int,
        transforms: List[TFM],
        target_transforms: List[TTFM],
        custom_dom_lbl: Optional[List[int]] = None,
    ):
        """Iterator that infinitely yield equal samples from several webdatasets.

        Args:
            urls (List[str | List[str]]): List of dataset URIs.
            num_workers (int): Number of workers.
                Will be divided internally by number of urls.
            batch_size (int): Batch size.
                Will be divided internally by number of urls.
            transforms (List[TFM]): Data transform for each dataset,
                must return `torch.Tensor`.
            target_transforms (List[TTFM]): Target transform for each dataset,
                must return `torch.Tensor`.
            custom_dom_lbl (Optional[List[int]], optional): Domain labels to return
                for each url. If None, domain labels is the index of each url.
        """
        assert len(urls) > 1
        assert len(transforms) == len(target_transforms) == len(urls)
        if custom_dom_lbl is not None:
            assert all([isinstance(el, int) for el in custom_dom_lbl])

        self.urls = urls
        self.num_urls = len(self.urls)

        self.num_workers = num_workers // self.num_urls
        self.batch_size = batch_size // self.num_urls
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.loaders = []
        for url, tfm, ttfm in zip(self.urls, self.transforms, self.target_transforms):
            if isinstance(url, list):
                exp_urls = []
                for u in url:
                    exp_urls.extend(list(braceexpand(u)))
                url = exp_urls
            self.loaders.append(self._setup_loader(url, tfm, ttfm))

        self.custom_dom_lbl = custom_dom_lbl

    def _setup_loader(self, u, tfm, ttfm):
        dataset = wds.WebDataset(u)
        dataset = (
            dataset.decode("pil")
            .to_tuple("png pickle")
            .map_tuple(tfm, ttfm)
            .batched(self.batch_size, partial=False)
            .repeat()
        )

        loader = wds.WebLoader(dataset, batch_size=None, num_workers=self.num_workers,)

        return iter(loader)

    def __iter__(self):
        while True:
            batch_inp = []
            batch_lbl = []
            batch_load_idx = []

            if self.custom_dom_lbl is not None:
                batch_dom_lbl = []

            for loader_idx, loader in enumerate(self.loaders):
                inp, lbl = next(loader)
                batch_inp.append(inp)
                batch_lbl.append(lbl)

                batch_load_idx.append(torch.LongTensor(inp.shape[0]).fill_(loader_idx))
                if self.custom_dom_lbl is not None:
                    batch_dom_lbl.append(
                        torch.LongTensor(inp.shape[0]).fill_(
                            self.custom_dom_lbl[loader_idx]
                        )
                    )

            batch_inp = torch.cat(batch_inp)
            batch_lbl = torch.cat(batch_lbl)
            batch_load_idx = torch.cat(batch_load_idx)
            if self.custom_dom_lbl is not None:
                batch_dom_lbl = torch.cat(batch_dom_lbl)

            if self.custom_dom_lbl is not None:
                yield batch_inp, batch_lbl, batch_load_idx, batch_dom_lbl
            else:
                yield batch_inp, batch_lbl, batch_load_idx


class MultiUnlimitedDataLoaderWB:
    def __init__(
        self,
        urls: List[Union[str, List[str]]],
        num_workers: int,
        batch_size: int,
        transforms: List[TFM],
        target_transforms: List[TTFM],
        custom_dom_lbl: Optional[List[int]] = None,
    ):
        """Iterator that infinitely yield equal samples from several webdatasets.

        Args:
            urls (List[str | List[str]]): List of dataset URIs.
            num_workers (int): Number of workers.
                Will be divided internally by number of urls.
            batch_size (int): Batch size.
                Will be divided internally by number of urls.
            transforms (List[TFM]): Data transform for each dataset,
                must return `torch.Tensor`.
            target_transforms (List[TTFM]): Target transform for each dataset,
                must return `torch.Tensor`.
            custom_dom_lbl (Optional[List[int]], optional): Domain labels to return
                for each url. If None, domain labels is the index of each url.
        """
        assert len(urls) > 1
        assert len(transforms) == len(target_transforms) == len(urls)
        if custom_dom_lbl is not None:
            assert all([isinstance(el, int) for el in custom_dom_lbl])

        self.urls = urls
        self.num_urls = len(self.urls)

        self.num_workers = num_workers // self.num_urls
        self.batch_size = batch_size // self.num_urls
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.loaders = []
        for url, tfm, ttfm in zip(self.urls, self.transforms, self.target_transforms):
            if isinstance(url, list):
                exp_urls = []
                for u in url:
                    exp_urls.extend(list(braceexpand(u)))
                url = exp_urls
            self.loaders.append(self._setup_loader(url, tfm, ttfm))

        self.custom_dom_lbl = custom_dom_lbl

    def _setup_loader(self, u, tfm, ttfm):
        dataset = wds.WebDataset(u)
        dataset = (
            dataset.decode("pil")
            .to_tuple("png pickle")
            .map(wbPreprocess)
            .map_tuple(tfm, ttfm)
            .batched(self.batch_size, partial=False)
            .repeat()
        )

        loader = wds.WebLoader(dataset, batch_size=None, num_workers=self.num_workers,)

        return iter(loader)

    def __iter__(self):
        while True:
            batch_inp = []
            batch_lbl = []
            batch_load_idx = []

            if self.custom_dom_lbl is not None:
                batch_dom_lbl = []

            for loader_idx, loader in enumerate(self.loaders):
                inp, lbl = next(loader)
                batch_inp.append(inp)
                batch_lbl.append(lbl)
                batch_load_idx.append(torch.LongTensor(inp.shape[0]).fill_(loader_idx))
                if self.custom_dom_lbl is not None:
                    batch_dom_lbl.append(
                        torch.LongTensor(inp.shape[0]).fill_(
                            self.custom_dom_lbl[loader_idx]
                        )
                    )

            batch_inp = torch.cat(batch_inp)
            batch_lbl = torch.cat(batch_lbl)
            batch_load_idx = torch.cat(batch_load_idx)
            if self.custom_dom_lbl is not None:
                batch_dom_lbl = torch.cat(batch_dom_lbl)

            if self.custom_dom_lbl is not None:
                yield batch_inp, batch_lbl, batch_load_idx, batch_dom_lbl
            else:
                yield batch_inp, batch_lbl, batch_load_idx


class MultiUnlimitedDataLoaderRR(MultiUnlimitedDataLoader):
    def __init__(
        self,
        urls: List[Union[str, List[str]]],
        num_workers: int,
        batch_size: int,
        transforms: List[TFM],
        target_transforms: List[TTFM],
    ):
        """Iterator that infinitely cycles samples from several webdatasets.

        Args:
            urls (List[str | List[str]]): List of dataset URIs.
            num_workers (int): Number of workers.
                Will be divided internally by number of urls.
            batch_size (int): Batch size.
                Will be divided internally by number of urls.
            transforms (List[TFM]): Data transform for each dataset,
                must return `torch.Tensor`.
            target_transforms (List[TTFM]): Target transform for each dataset,
                must return `torch.Tensor`.
        """
        super().__init__(urls, num_workers, batch_size, transforms, target_transforms)

    def __iter__(self):
        while True:
            for loader_idx, loader in enumerate(self.loaders):
                inp, lbl = next(loader)
                dom = torch.LongTensor(inp.shape[0]).fill_(loader_idx)

                yield inp, lbl, dom


def wbPreprocess(src):
    image, label = src
    probs = 0.5
    wb = wbAug.WBEmulator()
    mfs = wb.computeMappingFunc(image, 5)
    ind = np.random.randint(len(mfs))
    n = image
    if random() < probs:
        n = wbAug.changeWB(image, mfs[ind])  # pylint: disable=invalid-sequence-index
    return (n, label)
