from collections import deque
import numpy as np


__all__ = ["setup_comm", "average_distributed_deques",
           "popdist_root", "distributed_barrier"]


DISTRIBUTED_ROOT = 0


def popdist_root(args):
    if args.use_popdist:
        return args.popdist_rank == DISTRIBUTED_ROOT
    return True


DISTRIBUTED_COMM = None


def setup_comm(comm):
    global DISTRIBUTED_COMM
    DISTRIBUTED_COMM = comm


def _get_comm():
    global DISTRIBUTED_COMM
    if DISTRIBUTED_COMM is None:
        raise RuntimeError(
            "Distributed Commumication not setup. Please run setup_comm(MPI.COMM_WORLD) first. "
            "See https://mpi4py.readthedocs.io/ for details on MPI.COMM_WORLD.")
    return DISTRIBUTED_COMM


def distributed_barrier():
    _get_comm().barrier()


def average_distributed_deques(local_deque: deque, N: int = None) -> deque:
    comm = _get_comm()
    size = comm.Get_size()
    rank = comm.Get_rank()

    if N is None:
        N = len(local_deque)
    N = min(len(local_deque), N)
    _N = len(local_deque) - N

    sendbuf = np.array(local_deque, np.float32)[_N:]
    recvbuf = np.empty([size, sendbuf.size], np.float32)
    comm.Gather(sendbuf, recvbuf, root=DISTRIBUTED_ROOT)
    if rank == DISTRIBUTED_ROOT:
        avg = np.average(recvbuf, axis=0)
        for i, result in zip(range(_N, len(local_deque)), avg):
            local_deque[i] = result
    return local_deque


def sum_distributed_data(data: int) -> int:
    comm = _get_comm()
    size = comm.Get_size()
    rank = comm.Get_rank()

    sendbuf = np.array([data])
    recvbuf = np.empty([size, 1], int)
    comm.Allgather(sendbuf, recvbuf)
    data = np.sum(recvbuf, axis=0)

    return data
