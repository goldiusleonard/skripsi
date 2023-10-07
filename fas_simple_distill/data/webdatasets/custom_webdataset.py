#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import os

from webdataset import shardcache, tariterators
from webdataset.utils import lookup_sym, safe_eval
from webdataset.handlers import reraise_exception
from webdataset.shardlists import PytorchShardList, MultiShardSample

default_cache_dir = os.path.expanduser(os.environ.get("WEBDATASET_CACHE", ""))
default_cache_name = lookup_sym(
    os.environ.get("WEBDATASET_CACHE_NAME", "shard_uuid"), ".shardcache".split()
)
default_cache_verbose = int(safe_eval(os.environ.get("WEBDATASET_CACHE_VERBOSE", "1")))
default_cache_size = int(
    float(safe_eval(os.environ.get("WEBDATASET_CACHE_SIZE", "1e15")))
)


class NotSupportedError(Exception):
    pass


def WebDataset(
    urls,
    epoch_shuffle=False,
    shuffle=True,
    split_by_node=True,
    split_by_worker=True,
    cache_dir=default_cache_dir,
    cache_size=default_cache_size,
    cache_name=default_cache_name,
    cache_verbose=default_cache_verbose,
    handler=reraise_exception,
    repeat=False,
):
    """Return a pipeline for WebDataset-style data files.

    This is a convenience function for constructing a partial pipeline
    that reads from a set of sharded tar files, extracts the individual
    files, and groups them together into samples (dictionaries).

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    The recommended way of specifying novel ways of splitting shards is
    via writing a new shardlist class.

    :param urls: the source URLs: a string, a list, or an IterableDataset
    :param epoch_shuffle: assigns different shards to different nodes on each epoch
    :param shuffle: shuffle samples before iterating
    :param split_by_node: split shards by node if True
    :param split_by_worker: split shards by worker if True
    :param cache_dir: when set, caches shards in this directory
    :param cache_size: when set, specifies a maximum size for the shard cache
    :param cache_name: when set, specifies how shards should be named in the cache
    :param cache_verbose: when set, prints information about caching
    :param handler: an error handler
    :param repeat: repeat infinitely if True
    """
    if isinstance(urls, str) and urls.endswith(".ds.yml"):
        raise NotSupportedError
    if isinstance(urls, str):
        if urls.endswith(".shards.yml"):
            urls = MultiShardSample(urls)
        result = PytorchShardList(
            urls,
            epoch_shuffle=epoch_shuffle,
            shuffle=shuffle,
            split_by_worker=split_by_worker,
            split_by_node=split_by_node,
        )
    elif isinstance(urls, list):
        result = PytorchShardList(
            urls,
            epoch_shuffle=epoch_shuffle,
            shuffle=shuffle,
            split_by_worker=split_by_worker,
            split_by_node=split_by_node,
        )
    else:
        raise NotSupportedError
    result = result.then(tariterators.url_opener, handler=handler)
    if cache_dir != "":
        result = result.then(
            shardcache.cache_shards,
            cache_dir=cache_dir,
            cache_size=cache_size,
            cache_name=cache_name,
            verbose=cache_verbose,
        )
    result = result.then(tariterators.tar_file_expander, handler=handler)
    result = result.then(tariterators.group_by_keys)
    if repeat:
        result = result.repeat()
    return result
