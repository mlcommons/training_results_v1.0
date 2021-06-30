#include "common.hpp"
#include "data_reader.hpp"
#include "data_readers/async_reader/async_reader.hpp"
#include "data_readers/async_reader/async_reader_common.hpp"
#include "data_readers/async_reader/split_label_dense_sparse.hpp"
#include "tensor2.hpp"

namespace {
using namespace HugeCTR;

class RawPtrWrapper : public HugeCTR::TensorBuffer2 {
 public:
  RawPtrWrapper(void* ptr) : ptr_(ptr) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
};

class RawPtrBuffer : public HugeCTR::TensorBuffer2 {
 public:
  RawPtrBuffer(size_t size_bytes) { CK_CUDA_THROW_(cudaMalloc(&ptr_, size_bytes)); }
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }
  ~RawPtrBuffer() override { cudaFree(ptr_); }

 private:
  void* ptr_;
};

class DerivedPtrBuffer : public HugeCTR::TensorBuffer2 {
 public:
  DerivedPtrBuffer(void* ptr, const std::shared_ptr<RawPtrBuffer>& buffer)
      : ptr_(ptr), buffer_(buffer) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
  std::shared_ptr<RawPtrBuffer> buffer_;
};
}  // namespace

namespace HugeCTR {

template <typename SparseType>
class AsyncReader : public IDataReader {
  using LabelType = float;
  using InputType = int;

 public:
  // Default params: num_threads = num_local_gpus, io_block_size = 512000, io_depth = 2,
  // io_alignment = 512
  AsyncReader(std::string fname, size_t batch_size, size_t label_dim, size_t dense_dim,
              std::vector<DataReaderSparseParam>& params, bool mixed_precision,
              const std::shared_ptr<ResourceManager>& resource_manager, int num_threads,
              int num_batches_per_thread, size_t io_block_size, int io_depth, int io_alignment)
      : resource_manager_(resource_manager),
        mixed_precision_(mixed_precision),
        batch_size_(batch_size) {
    assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
    assert(params.size() == 1);
    static_assert(sizeof(LabelType) == sizeof(InputType));

    batch_size_per_dev_ = batch_size_ / resource_manager_->get_global_gpu_count();

    size_t sparse_dim = params[0].slot_num;
    sample_size_items_ =
        label_dim + dense_dim + sparse_dim * (sizeof(SparseType) / sizeof(InputType));
    size_t batch_size_bytes = sample_size_items_ * sizeof(InputType) * batch_size;
    // for (auto p : params) {
    //   assert(p.max_nnz == 1);
    // }

    reader_impl_ = std::make_unique<AsyncReaderImpl>(
        fname, batch_size_bytes, resource_manager.get(), num_threads, num_batches_per_thread,
        io_block_size, io_depth, io_alignment);

    for (uint32_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      auto local_gpu = resource_manager_->get_local_gpu(i);
      CudaDeviceContext ctx(local_gpu->get_device_id());

      auto my_dev_id = resource_manager_->get_gpu_global_id_from_local_id(i);

      auto label_buffer_ =
          std::make_shared<RawPtrBuffer>(batch_size * label_dim * sizeof(LabelType));
      label_tensors_.emplace_back(
          Tensor2<LabelType>({batch_size, label_dim}, label_buffer_).shrink());

      auto label_buffer_offset = my_dev_id * (batch_size_per_dev_ * label_dim);
      auto label_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
          ((LabelType*)(label_buffer_->get_ptr()) + label_buffer_offset), label_buffer_);
      label_tensors_per_dev_.emplace_back(
          Tensor2<LabelType>({batch_size_per_dev_, label_dim}, label_buffer_per_dev).shrink());

      if (mixed_precision_) {
        auto dense_buffer_ =
            std::make_shared<RawPtrBuffer>(batch_size * dense_dim * sizeof(__half));
        dense_tensors_.emplace_back(
            Tensor2<__half>({batch_size, dense_dim}, dense_buffer_).shrink());

        auto dense_buffer_offset = my_dev_id * (batch_size_per_dev_ * dense_dim);
        auto dense_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
            ((__half*)dense_buffer_->get_ptr() + dense_buffer_offset), dense_buffer_);
        dense_tensors_per_dev_.emplace_back(
            Tensor2<__half>({batch_size_per_dev_, dense_dim}, dense_buffer_per_dev).shrink());

      } else {
        auto dense_buffer_ = std::make_shared<RawPtrBuffer>(batch_size * dense_dim * sizeof(float));
        dense_tensors_.emplace_back(
            Tensor2<float>({batch_size, dense_dim}, dense_buffer_).shrink());

        auto dense_buffer_offset = my_dev_id * (batch_size_per_dev_ * dense_dim);
        auto dense_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
            ((float*)dense_buffer_->get_ptr() + dense_buffer_offset), dense_buffer_);
        dense_tensors_per_dev_.emplace_back(
            Tensor2<float>({batch_size_per_dev_, dense_dim}, dense_buffer_per_dev).shrink());
      }

      auto sparse_buffer_ =
          std::make_shared<RawPtrBuffer>(batch_size * sparse_dim * sizeof(SparseType));
      sparse_tensors_.emplace_back(
          Tensor2<SparseType>({batch_size, sparse_dim}, sparse_buffer_).shrink());
    }
  }

  long long read_a_batch_to_device() override {
    auto batch = reader_impl_->get_batch();
    if (batch.size_bytes == 0) {
      reader_impl_->reset();
      reader_impl_->load_async();
      batch = reader_impl_->get_batch();
    }

    int num_local_gpus = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(num_local_gpus)
    for (int i = 0; i < num_local_gpus; i++) {
      auto local_gpu = resource_manager_->get_local_gpu(i);
      CudaCPUDeviceContext ctx(local_gpu->get_device_id());

      current_batch_size_ = batch.size_bytes / (sample_size_items_ * sizeof(InputType));
      auto ptr_wrap =
          std::make_shared<RawPtrWrapper>(reinterpret_cast<InputType*>(batch.dev_data[i]));

      if (mixed_precision_) {
        split_3_way<__half, SparseType>(
            Tensor2<LabelType>::stretch_from(label_tensors_[i]),
            Tensor2<__half>::stretch_from(dense_tensors_[i]),
            Tensor2<SparseType>::stretch_from(sparse_tensors_[i]),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            local_gpu->get_stream());
      } else {
        split_3_way<float, SparseType>(
            Tensor2<LabelType>::stretch_from(label_tensors_[i]),
            Tensor2<float>::stretch_from(dense_tensors_[i]),
            Tensor2<SparseType>::stretch_from(sparse_tensors_[i]),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            local_gpu->get_stream());
      }

      CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_stream()));
    }

    reader_impl_->finalize_batch();
    return current_batch_size_;
  }

  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<SparseType>::get_type();
  };
  long long read_a_batch_to_device_delay_release() override { return read_a_batch_to_device(); }
  long long get_current_batchsize_per_device(size_t local_id) override {
    long long batchsize_per_device = batch_size_ / resource_manager_->get_global_gpu_count();
    size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
    long long remain_samples = current_batch_size_ - global_id * batchsize_per_device;
    if (remain_samples >= batchsize_per_device) {
      return batchsize_per_device;
    } else if (remain_samples > 0) {
      return remain_samples;
    } else {
      return 0;
    }
  }
  bool is_started() const override { return reader_impl_->is_currently_loading(); }
  void start() override { reader_impl_->load_async(); }
  std::vector<TensorBag2> get_label_tensors() const override { return label_tensors_per_dev_; }
  std::vector<TensorBag2> get_dense_tensors() const override { return dense_tensors_per_dev_; }
  std::vector<TensorBag2> get_row_offsets_tensors() const override { return {}; };
  std::vector<TensorBag2> get_value_tensors() const override { return sparse_tensors_; };

  ~AsyncReader() override {}

  void ready_to_collect() override {}
  void create_drwg_norm(std::string file_list, Check_t check_type,
                        bool start_reading_from_beginning = true) override {}
  void create_drwg_raw(std::string file_name, long long num_samples,
                       const std::vector<long long> slot_offset, bool float_label_dense,
                       bool data_shuffle, bool start_reading_from_beginning = true) override {}
#ifndef DISABLE_CUDF
  void create_drwg_parquet(std::string file_list, const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true) override {}
#endif
  void set_file_list_source(std::string file_list = std::string()) override {}

 public:
  const std::shared_ptr<ResourceManager> resource_manager_;
  std::unique_ptr<AsyncReaderImpl> reader_impl_;
  size_t sample_size_items_, current_batch_size_;
  bool mixed_precision_;
  size_t batch_size_, batch_size_per_dev_;

  std::vector<TensorBag2> label_tensors_;
  std::vector<TensorBag2> dense_tensors_;
  std::vector<TensorBag2> label_tensors_per_dev_;
  std::vector<TensorBag2> dense_tensors_per_dev_;
  std::vector<TensorBag2> sparse_tensors_;
};

}  // namespace HugeCTR
