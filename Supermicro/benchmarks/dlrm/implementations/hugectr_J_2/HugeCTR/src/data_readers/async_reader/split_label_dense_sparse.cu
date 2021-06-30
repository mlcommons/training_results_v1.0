#include <common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>

namespace HugeCTR {

// Sparse pointer should be casted to int* when calling this kernel
template <typename DenseType, typename SparseType>
__global__ void split_kernel_3_way__(int batch_size, float* label_ptr, int label_dim,
                                     DenseType* dense_ptr, int dense_dim, int* sparse_ptr,
                                     int sparse_dim, const int* label_dense_sparse) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int sample_size_int =
      (label_dim + dense_dim + sparse_dim * (sizeof(SparseType) / sizeof(int)));

  if (idx < batch_size * sample_size_int) {
    const int in_col = idx % sample_size_int;
    const int in_row = idx / sample_size_int;
    const int out_row = in_row;
    if (in_col < label_dim) {
      const int out_col = in_col;
      int label = label_dense_sparse[idx];
      label_ptr[out_row * label_dim + out_col] = label;
    } else if (in_col < label_dim + dense_dim) {
      const int out_col = in_col - label_dim;
      int dense = label_dense_sparse[idx];
      dense_ptr[out_row * dense_dim + out_col] =
          logf(dense + 1.f);  // TODO : FIXME move to data preprocessing
    } else {
      const int out_col = in_col - label_dim - dense_dim;
      sparse_ptr[out_row * sparse_dim + out_col] = label_dense_sparse[idx];
    }
  }
  return;
}

template <typename DenseType, typename SparseType>
void split_3_way(Tensor2<float> label_tensor, Tensor2<DenseType> dense_tensor,
                 Tensor2<SparseType> sparse_tensor, Tensor2<int> label_dense_sparse_buffer,
                 cudaStream_t stream) {
  assert(label_tensor.get_dimensions()[0] == dense_tensor.get_dimensions()[0]);
  assert(label_tensor.get_dimensions()[0] == sparse_tensor.get_dimensions()[0]);
  // assert(label_tensor.get_num_elements() + dense_tensor.get_num_elements() +
  //            sparse_tensor.get_num_elements() ==
  //        label_dense_sparse_buffer.get_num_elements());

  const int batch_size = label_dense_sparse_buffer.get_dimensions()[0];
  const int label_dim = label_tensor.get_dimensions()[1];
  const int dense_dim = dense_tensor.get_dimensions()[1];
  const int sparse_dim = sparse_tensor.get_dimensions()[1];

  const int BLOCK_DIM = 256;
  const int GRID_DIM = (label_dense_sparse_buffer.get_num_elements() - 1) / BLOCK_DIM + 1;
  assert(dense_dim >= 0 || "dense_dim should be >= 0");

  if (GRID_DIM > 0) {
    split_kernel_3_way__<DenseType, SparseType><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        batch_size, label_tensor.get_ptr(), label_dim, dense_tensor.get_ptr(), dense_dim,
        reinterpret_cast<int*>(sparse_tensor.get_ptr()), sparse_dim,
        label_dense_sparse_buffer.get_ptr());
  }

  return;
}

template void split_3_way<float, uint32_t>(Tensor2<float> label_tensor, Tensor2<float> dense_tensor,
                                           Tensor2<uint32_t> sparse_tensor,
                                           Tensor2<int> label_dense_sparse_buffer,
                                           cudaStream_t stream);
template void split_3_way<__half, uint32_t>(Tensor2<float> label_tensor,
                                            Tensor2<__half> dense_tensor,
                                            Tensor2<uint32_t> sparse_tensor,
                                            Tensor2<int> label_dense_sparse_buffer,
                                            cudaStream_t stream);

template void split_3_way<float, long long>(Tensor2<float> label_tensor,
                                            Tensor2<float> dense_tensor,
                                            Tensor2<long long> sparse_tensor,
                                            Tensor2<int> label_dense_sparse_buffer,
                                            cudaStream_t stream);
template void split_3_way<__half, long long>(Tensor2<float> label_tensor,
                                             Tensor2<__half> dense_tensor,
                                             Tensor2<long long> sparse_tensor,
                                             Tensor2<int> label_dense_sparse_buffer,
                                             cudaStream_t stream);

}  // namespace HugeCTR
