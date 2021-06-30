import os
from time import time

import numpy as np
from mpi4py import MPI
from mxnet import nd

def distribute_mpiranks(local_rank, local_size, size, nodes_for_eval, gpu_per_node):
    # assign top "nodes_for_eval" nodes for evaluation. Rest of the nodes go to training
    total_ranks = list(range(size))
    train_ranks = total_ranks[:size - nodes_for_eval * gpu_per_node]
    eval_ranks = train_ranks
    transfer_ranks = []
    if nodes_for_eval:
        eval_ranks = total_ranks[size - nodes_for_eval * gpu_per_node:]
        # print(f"Training ranks {train_ranks} \nEval ranks {eval_ranks}")
        #transfer_ranks = [train_ranks[0], eval_ranks[0]]
        # Form multiple transfer_rank groups, by local_rank
        transfer_ranks = [train_ranks[local_rank], *[x for x in eval_ranks if x % local_size == local_rank]]
    assert train_ranks, "Training ranks list is empty"
    assert eval_ranks, "Evaluation ranks list is empty"
    # print(f"TRANSFER RANKS {transfer_ranks}")
    return train_ranks, eval_ranks, transfer_ranks


def get_group_comm(comm, ranks):
    # Create a grouped mpi communicator with the ranks
    # assert len(ranks) > 0, "cannot create group as ranks is empty"
    xcomm = None
    if ranks:
        xgroup = comm.group.Incl(ranks)
        xcomm = comm.Create_group(xgroup)

    return xcomm


def sync_training_and_evaluation(flags, global_comm, eval_comm, transfer_comm,
                                 rank, model, train_ranks, eval_ranks, transfer_ranks,
                                 cycle, stop_training, ctx):

    # Let training threads know if evaluation has reached target
    # All reduce also acts as barrier to make sure parameter save is done
    local_stop_training = np.array([stop_training], dtype=np.int32)
    global_stop_training = np.zeros(1, dtype=np.int32)
    global_comm.Allreduce(local_stop_training, global_stop_training, MPI.SUM)

    start = time()
    filename = os.path.join(flags.network_dir, f'model_{cycle}.params')
    if flags.use_mpi_bcast:
        if rank in transfer_ranks:
            broadcast_model(model, transfer_comm, rank, eval_ranks)
    elif flags.use_mpi_transfer:
        if rank == train_ranks[0] or rank in eval_ranks:
            transfer_model(model, global_comm, eval_comm, rank, train_ranks[0], eval_ranks[0], eval_ranks)
    else:
        if rank == train_ranks[0]:
            model.save_parameters(filename)

    # Evaluation found end of training
    if global_stop_training != 0:
        stop_training = True
    else:
        if not flags.use_mpi_bcast and not flags.use_mpi_transfer:
            # load model for evaluation
            if rank in eval_ranks:
                if os.path.exists(filename):
                    model.load_parameters(filename, ctx=ctx)
                else:
                    raise Exception(f"rank {rank}: model does not exist for {cycle}")

    if rank == train_ranks[0]:
        print(f"rank {rank}: cycle = {cycle}: time to send the model = {time() - start}")
    if rank == eval_ranks[0]:
        print(f"rank {rank}: cycle = {cycle}: time to receive the model = {time() - start}")

    return stop_training, model


def broadcast_model(model, comm, rank, eval_ranks):
    params = model._collect_params_with_prefix()

    irequests = []
    result = {}
    for name, p in sorted(params.items()):
        if "dummy" in name:
            continue
        result[name] = p.data().asnumpy()
        irequests.append(comm.Ibcast(result[name], root=0))

    MPI.Request.waitall(irequests)

    if rank in eval_ranks:
        for name, p in sorted(params.items()):
            if "dummy" in name:
                continue
            params[name].set_data(result[name])


def transfer_model(model, global_comm, eval_comm, rank, source_rank, target_rank, eval_ranks):
    params = model._collect_params_with_prefix()

    irequests = []
    result = {}
    for idx, (name, p) in enumerate(sorted(params.items())):
        if "dummy" in name:
            continue
        data = p.data().asnumpy()
        if rank == source_rank:
            irequests.append(global_comm.Isend(data, dest=target_rank, tag=idx))
        elif rank == target_rank:
            result[name] = data
            irequests.append(global_comm.Irecv(result[name], source=source_rank, tag=idx))
        else:
            result[name] = data

    if rank == source_rank:
        MPI.Request.waitall(irequests)

    elif rank in eval_ranks:
        if rank == target_rank:
            MPI.Request.waitall(irequests)
        eval_comm.Barrier()
        for idx, (name, p) in enumerate(sorted(params.items())):
            if "dummy" in name or name not in result.keys():
                continue
            # data = p.data().asnumpy()
            eval_comm.Bcast(result[name], root=0)
            # params[name]._load_init(nd.array(result[name]), ctx, cast_dtype=False, dtype_source='current')
            params[name].set_data(result[name])
