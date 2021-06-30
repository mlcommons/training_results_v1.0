
#include "data_readers/async_reader/thread_async_reader.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <numeric>
#include <stdexcept>

#include "common.hpp"
#include "data_readers/async_reader/async_reader_common.hpp"
#include "data_readers/async_reader/broadcast.hpp"
#include "resource_manager.hpp"

namespace HugeCTR {

ThreadAsyncReader::ThreadAsyncReader(std::string fname, const ResourceManager* resource_mananager,
                                     size_t batch_size_bytes, int device_id, cudaStream_t stream,
                                     std::vector<size_t> batch_ids,
                                     std::vector<InternalBatchBuffer*> dest_buffers,
                                     int io_block_size, int io_alignment, int io_depth,
                                     size_t total_file_size)
    : batch_size_bytes_(batch_size_bytes),
      device_id_(device_id),
      stream_(stream),
      io_block_size_(io_block_size),
      io_depth_(io_depth),
      total_file_size_(total_file_size),
      batch_ids_(batch_ids),
      dest_buffers_(dest_buffers) {
#if (__cplusplus >= 201703L)
  static_assert(std::atomic<BufferStatus>::is_always_lock_free &&
                    std::atomic<WorkerStatus>::is_always_lock_free,
                "Compiler cannot use atomic enum class, need to change to int type");
#endif
  assert(io_block_size_ % io_alignment == 0);

  num_dest_buffers_ = dest_buffers_.size();

  fd_ = open(fname.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ == -1) {
    throw std::runtime_error("No such file: " + fname);
  };

  max_num_blocks_per_batch_ = batch_size_bytes_ / io_block_size_ + 2;
  for (auto buf : dest_buffers_) {
    assert((size_t)buf->raw_host_ptr % io_alignment == 0);
    CK_CUDA_THROW_(cudaMallocHost(&buf->raw_host_ptr, max_num_blocks_per_batch_ * io_block_size_));
    CK_CUDA_THROW_(cudaEventCreate(&buf->event, cudaEventDisableTiming));

    int num_local_gpus = buf->dev_data.size();
    auto alloc_duplicated = [num_local_gpus, this](void** ptr, size_t size) {
      CK_CUDA_THROW_(cudaMallocManaged(ptr, size));
      CK_CUDA_THROW_(cudaMemAdvise(*ptr, size, cudaMemAdviseSetReadMostly, device_id_));
    };
    alloc_duplicated((void**)&buf->dev_pointers, num_local_gpus * sizeof(float*));
    alloc_duplicated((void**)&buf->dev_p2p_accessible, num_local_gpus * sizeof(bool));

    for (int dev = 0; dev < num_local_gpus; dev++) {
      buf->dev_pointers[dev] = (float*)buf->dev_data[dev];
      buf->dev_p2p_accessible[dev] = resource_mananager->p2p_enabled(device_id, dev);
    }

    buf->io_reqs.resize(max_num_blocks_per_batch_);
    for (auto& req : buf->io_reqs) {
      req = new iocb;
    }
  }

  for (auto buf : dest_buffers_) {
    buf->status.store(BufferStatus::IOReady);
  }
}

void ThreadAsyncReader::load() {
  size_t num_batches = batch_ids_.size();
  size_t processed = 0;
  std::vector<size_t> id_per_host_buffer(num_dest_buffers_);
  std::iota(id_per_host_buffer.begin(), id_per_host_buffer.end(), 0);

  status_.store(WorkerStatus::OK);

  ioctx_ = 0;
  if (io_queue_init(io_depth_, &ioctx_) < 0) {
    throw std::runtime_error("io_setup failed");
  }

  while (processed != num_batches && status_.load() != WorkerStatus::Terminate) {
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        try_submit_io(batch_ids_[id_per_host_buffer[i]], i);
      }
    }
    wait_io();
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        try_submit_upload(batch_ids_[id_per_host_buffer[i]], i);
      }
    }
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        if (check_completion(batch_ids_[id_per_host_buffer[i]], i)) {
          processed++;
          id_per_host_buffer[i] += num_dest_buffers_;
        }
      }
    }
    // printf("processed: %lu / %lu\n", processed, num_batches);
  }

  if (io_destroy(ioctx_) < 0) {
    throw std::runtime_error("io_destroy failed");
  }

  if (status_.load() != WorkerStatus::Terminate) {
    for (int i = 0; i < num_dest_buffers_; i++) {
      BufferStatus expected = BufferStatus::IOReady;
      while (!dest_buffers_[i]->status.compare_exchange_weak(expected, BufferStatus::Finished)) {
        expected = BufferStatus::IOReady;
      }
      // printf("finalized buffer %d, stat %d\n", i, (int)dest_buffers_[i]->status.load());
    }
  }
}

void ThreadAsyncReader::try_submit_io(size_t batch_id, int io_id) {
  auto& buffer = dest_buffers_[io_id];
  if (buffer->status.load() != BufferStatus::IOReady) {
    return;
  }

  buffer->status.store(BufferStatus::IOInProcess);

  size_t req_beg_offset = batch_id * batch_size_bytes_;
  size_t req_end_offset = std::min((batch_id + 1) * batch_size_bytes_, total_file_size_);
  size_t raw_beg_offset = (req_beg_offset / io_block_size_) * io_block_size_;
  size_t raw_end_offset = ((req_end_offset + io_block_size_ - 1) / io_block_size_) * io_block_size_;
  size_t num_blocks = (raw_end_offset - raw_beg_offset) / io_block_size_;
  assert(num_blocks <= (size_t)max_num_blocks_per_batch_);

  buffer->id = batch_id;
  buffer->num_outstanding_reqs = num_blocks;
  buffer->size = req_end_offset - req_beg_offset;
  buffer->host_data = buffer->raw_host_ptr + (req_beg_offset - raw_beg_offset);
  assert(buffer->size % sizeof(float) == 0);

  for (size_t block = 0; block < num_blocks; block++) {
    auto req = buffer->io_reqs[block];

    io_prep_pread(req, fd_, buffer->raw_host_ptr + io_block_size_ * block, io_block_size_,
                  raw_beg_offset + io_block_size_ * block);
    req->data = (void*)buffer;
  }

  int ret = io_submit(ioctx_, num_blocks, buffer->io_reqs.data());
  if (ret < 0) {
    throw std::runtime_error("io_submit failed");
  }
}

void ThreadAsyncReader::wait_io() {
  timespec timeout = {0, 10'000l};
  io_event events[max_num_blocks_per_batch_];
  int num_completed =
      io_getevents(ioctx_, max_num_blocks_per_batch_, max_num_blocks_per_batch_, events, &timeout);
  if (num_completed < 0) {
    throw std::runtime_error("io_getevents failed");
  }

  for (int b = 0; b < num_completed; b++) {
    auto req = events[b].obj;
    auto buffer = (InternalBatchBuffer*)req->data;
    buffer->num_outstanding_reqs--;
    assert(buffer->num_outstanding_reqs >= 0);
  }
}

void ThreadAsyncReader::try_submit_upload(size_t batch_id, int io_id) {
  auto& buffer = dest_buffers_[io_id];
  if (buffer->status.load() != BufferStatus::IOInProcess || buffer->num_outstanding_reqs > 0) {
    return;
  }

  CK_CUDA_THROW_(cudaMemcpyAsync(buffer->dev_data[device_id_], buffer->host_data, buffer->size,
                                 cudaMemcpyHostToDevice, stream_));

  broadcast(buffer->dev_pointers, buffer->dev_p2p_accessible, buffer->size / sizeof(float),
            buffer->dev_data.size(), device_id_, stream_);

  // There is no real need to make eventRecord atomic (wrt stream) with the rest,
  //  we only care that eventRecord is AFTER the memcpy and the broadcast
  CK_CUDA_THROW_(cudaEventRecord(buffer->event, stream_));
  buffer->status.store(BufferStatus::UploadInProcess);
}

bool ThreadAsyncReader::check_completion(size_t batch_id, int io_id) {
  auto& buffer = dest_buffers_[io_id];
  if (buffer->status.load() != BufferStatus::UploadInProcess) {
    return false;
  }

  auto res = cudaEventQuery(buffer->event);
  if (res == cudaSuccess) {
    buffer->status.store(BufferStatus::ReadReady);
    return true;
  }
  if (res == cudaErrorNotReady) {
    return false;
  }
  CK_CUDA_THROW_(res);
  return false;
}

void ThreadAsyncReader::reset() {
  status_.store(WorkerStatus::Terminate);
  for (auto buf : dest_buffers_) {
    buf->status.store(BufferStatus::IOReady);
  }
}

ThreadAsyncReader::~ThreadAsyncReader() {}

}  // namespace HugeCTR
