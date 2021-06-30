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


def get_seed(master_seed, spatial_group_size, spatial_loader):
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

    if spatial_loader:
        return worker_seeds[hvd.rank()]
    return worker_seeds[hvd.rank() // spatial_group_size]


def get_rnd_scratch_space(network_dir):
    return os.path.join(network_dir, os.environ.get("SLURM_JOBID", str(random.randint(9999, 99999))))


def cleanup_scratch_space(scratch_space, nodes_for_eval, global_rank):
    #create an empty scratch space for models
    if global_rank == 0 and nodes_for_eval:
        try:
            if os.path.exists(scratch_space):
                shutil.rmtree(scratch_space)
            os.makedirs(scratch_space)
        except:
            print(f"CLEANING SCRATCH SPACE FAILED. PATH {scratch_space} IS INACCESSIBLE.")
            pass

