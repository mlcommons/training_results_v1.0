#include "data_readers/async_reader/async_reader.hpp"

#include <cuda_runtime.h>
#include <numa.h>
#include <nvml.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>

#include "common.hpp"
#include "resource_manager.hpp"
#include "utils.hpp"

namespace {
size_t get_file_size(std::string fname) {
  std::ifstream in(fname, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}
}  // namespace

namespace HugeCTR {

AsyncReaderImpl::AsyncReaderImpl(std::string fname, size_t batch_size_bytes,
                                 const ResourceManager* resource_manager, int num_threads,
                                 int num_batches_per_thread, size_t io_block_size, int io_depth,
                                 int io_alignment)
    :

      fname_(fname),
      batch_size_bytes_(batch_size_bytes),
      resource_manager_(resource_manager),
      num_devices_(resource_manager_->get_local_gpu_count()),
      num_threads_(num_threads),
      num_batches_per_thread_(num_batches_per_thread),
      io_block_size_(io_block_size),
      io_depth_(io_depth),
      io_alignment_(io_alignment),
      queue_id_(0),
      buffers_(num_threads_ * num_batches_per_thread),
      thread_batch_ids_(num_threads_),
      local_readers_(num_threads_) {
  total_file_size_ = get_file_size(fname);
  num_batches_ = (total_file_size_ + batch_size_bytes_ - 1) / batch_size_bytes;
  batch_ids_.resize(num_batches_);
  std::iota(batch_ids_.begin(), batch_ids_.end(), 0);

  for (auto& buf : buffers_) {
    buf.dev_data.resize(num_devices_);
    for (int id = 0; id < num_devices_; id++) {
      auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
      CudaDeviceContext ctx(device_id);
      CK_CUDA_THROW_(cudaMalloc(&buf.dev_data[id], batch_size_bytes_));
    }
  }

  streams_.resize(num_devices_);
  for (int id = 0; id < num_devices_; id++) {
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaDeviceContext ctx(device_id);
    CK_CUDA_THROW_(cudaStreamCreateWithPriority(&streams_[id], cudaStreamNonBlocking, 100));
  }

  // For correct perf benchmarking create the thread readers upfront
  create_workers();
}

void AsyncReaderImpl::create_workers() {
  // Use round-robin distribution
  const int chunk = num_batches_per_thread_;
  for (size_t i = 0; i < num_batches_; i++) {
    int thid = (int)(i % (chunk * num_threads_)) / chunk;
    thread_batch_ids_[thid].push_back(batch_ids_[i]);
    // printf("thread %d got buffer %lu\n", thid, i);
  }

  for (int thid = 0; thid < num_threads_; thid++) {
    threads_.emplace_back(std::thread([thid, this]() {
      int raw_id = (num_devices_ * thid) / num_threads_;
      int device_id = resource_manager_->get_local_gpu(raw_id)->get_device_id();
      CudaCPUDeviceContext ctx(device_id);

      std::vector<InternalBatchBuffer*> thread_batch_buffers(num_batches_per_thread_);
      for (int i = 0; i < num_batches_per_thread_; i++) {
        thread_batch_buffers[i] = &buffers_[thid * num_batches_per_thread_ + i];
      }

      local_readers_[thid] = std::make_unique<ThreadAsyncReader>(
          fname_, resource_manager_, batch_size_bytes_, raw_id, streams_[raw_id],
          thread_batch_ids_[thid], thread_batch_buffers, io_block_size_, io_alignment_, io_depth_,
          total_file_size_);
    }));
  }

  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

bool AsyncReaderImpl::is_currently_loading() { return !threads_.empty(); }

void AsyncReaderImpl::load_async() {
  if (is_currently_loading()) {
    throw std::runtime_error("load_async() is called before the previous load_async finished!");
  }

  for (int thid = 0; thid < num_threads_; thid++) {
    threads_.emplace_back(std::thread([thid, this]() {
      int raw_id = (num_devices_ * thid) / num_threads_;
      int device_id = resource_manager_->get_local_gpu(raw_id)->get_device_id();
      CudaCPUDeviceContext ctx(device_id);

      local_readers_[thid]->load();
    }));
  }
}

BatchDesc AsyncReaderImpl::get_batch() {
  if (!is_currently_loading()) {
    throw std::runtime_error(
        "Requested a batch from a file that is not being loaded. Please call load_async() first!");
  }

  for (int attempt = 0; attempt < (int)buffers_.size(); attempt++) {
    last_buffer_ = &buffers_[queue_id_];

    while (last_buffer_->status.load() != BufferStatus::Finished) {
      if (last_buffer_->status.load() == BufferStatus::ReadReady) {
        // printf("read batch %lu size %lu\n", last_buffer_->id, last_buffer_->size);
        return {last_buffer_->size, last_buffer_->dev_data};
      }
    }

    queue_id_ = (queue_id_ + 1) % buffers_.size();
  }

  return {0, std::vector<char*>(0)};
}

void AsyncReaderImpl::finalize_batch() {
  // Don't update status of finished buffers
  // printf("finalizing batch %lu\n", last_buffer_->id);
  BufferStatus expected = BufferStatus::ReadReady;
  last_buffer_->status.compare_exchange_strong(expected, BufferStatus::IOReady);
  // printf("done batch %lu\n", last_buffer_->id);
  queue_id_ = (queue_id_ + 1) % buffers_.size();
}

void AsyncReaderImpl::reset() {
  for (auto& reader : local_readers_) {
    reader->reset();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
  queue_id_ = 0;
}

AsyncReaderImpl::~AsyncReaderImpl() { reset(); }

}  // namespace HugeCTR
