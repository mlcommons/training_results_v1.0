#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <vector>

struct iocb;

namespace HugeCTR {

enum class BufferStatus : int {
  IOReady = 0,
  IOInProcess = 1,
  UploadInProcess = 2,
  ReadReady = 3,
  Finished = 4
};

struct InternalBatchBuffer {
  size_t id;
  size_t size;

  std::vector<char*> dev_data;
  float** dev_pointers;
  bool* dev_p2p_accessible;
  char* raw_host_ptr = nullptr;
  char* host_data;

  std::atomic<BufferStatus> status;
  std::vector<iocb*> io_reqs;
  int num_outstanding_reqs;
  cudaEvent_t event;

  // Following the rule of 5 just in case
  // Only need the destructor here
  InternalBatchBuffer() { status.store(BufferStatus::IOReady); };
  InternalBatchBuffer(InternalBatchBuffer const& other) = delete;
  InternalBatchBuffer& operator=(InternalBatchBuffer const& other) = delete;

  InternalBatchBuffer(InternalBatchBuffer&& other) = default;
  InternalBatchBuffer& operator=(InternalBatchBuffer&& other) = default;

  ~InternalBatchBuffer() {
    for (auto ptr : dev_data) {
      cudaFree(ptr);
    }
    cudaFreeHost(raw_host_ptr);
    cudaFree(dev_pointers);
    cudaFree(dev_p2p_accessible);
  }
};

struct BatchDesc {
  size_t size_bytes;
  std::vector<char*> dev_data;
};

}  // namespace HugeCTR