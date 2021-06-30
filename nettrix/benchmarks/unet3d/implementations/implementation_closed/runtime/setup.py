import os
import random
import numpy as np
import mxnet as mx
import horovod.mxnet as hvd
import shutil, pdb


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)


def check_flags(flags, world_size):
    assert not flags.amp or not flags.static_cast, "amp and static_cast are not compatible"
    assert world_size >= flags.spatial_group_size, "World size is smaller than SpatialGroupSize"
    if flags.spatial_group_size > 1:
        assert flags.batch_size == 1, f"batch_size must be equal to 1, got {flags.batch_size}"
        assert flags.val_batch_size == 1, f"val_batch_size must be equal to 1, got {flags.val_batch_size}"


def set_flags(params):
    if not params.benchmark:
        os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    """
    if hvd.size() > 1:
        seeds_tensor = mx.ndarray.array(seeds, ctx=mx.gpu(hvd.local_rank()), dtype=np.int64)
        seeds_tensor = hvd.broadcast(seeds_tensor, root_rank=0, name="broadcast_seed")
        seeds = seeds_tensor.as_in_context(mx.cpu()).asnumpy().tolist()

    return seeds

def get_seed(master_seed, spatial_group_size, cached_loader, stick_to_shard):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    """
    if master_seed == -1:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        # initialize seeding RNG
        seeding_rng = random.Random(master_seed)
        worker_seeds = generate_seeds(seeding_rng, hvd.size())
        worker_seeds = broadcast_seeds(worker_seeds)
        if hvd.rank() == 0:
            print(f'Using random master seed: {master_seed}. Worker seeds {len(worker_seeds)}: {worker_seeds}')
    else:
        worker_seeds = [master_seed + (hvd.rank() // spatial_group_size)] * hvd.size()
        if hvd.rank() == 0:
            print(f'Using master seed from command line. Worker seeds {len(worker_seeds)}: {worker_seeds}')
    if cached_loader and not stick_to_shard:
        return worker_seeds[0]
    return worker_seeds[hvd.rank() // spatial_group_size]
