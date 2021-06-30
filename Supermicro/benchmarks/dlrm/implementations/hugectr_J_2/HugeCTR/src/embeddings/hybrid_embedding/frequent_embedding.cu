/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/update.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

namespace frequent_embedding_kernels {

__global__ void forward_model(const float* const* __restrict__ embedding_vectors_pointers,
                              float* embedding_vectors_out,
                              const uint32_t* __restrict__ network_cache_indices,
                              const uint32_t* __restrict__ network_cache_indices_offsets,
                              uint32_t num_instances, uint32_t network_id,
                              uint32_t num_frequent_per_model, uint32_t embedding_vec_size) {
  const uint32_t offset = __ldg(network_cache_indices_offsets + network_id + 1);
  const uint32_t num_network_cache_indices = __ldg(network_cache_indices_offsets + num_instances);

  for (uint32_t i = blockIdx.x; i < num_network_cache_indices; i += gridDim.x) {
    int vid = (i + offset) % num_network_cache_indices;

    uint32_t frequent_index = network_cache_indices[vid];
    uint32_t model_id = frequent_index / num_frequent_per_model;

    const float* embedding_vectors_in = embedding_vectors_pointers[model_id];

    uint32_t cache_location = frequent_index * embedding_vec_size + threadIdx.x;
    embedding_vectors_out[cache_location] = embedding_vectors_in[cache_location];
  }
}

template <typename dtype, typename emtype>
__global__ void forward_network(uint32_t embedding_vec_size, uint32_t global_sample_index_base,
                                const uint32_t* frequent_sample_indices,  // local
                                const dtype* samples,                     // global
                                const dtype* category_frequent_index,     // global
                                const float* frequent_embedding_vectors,  // global
                                emtype* interaction_layer_input,
                                const uint32_t* d_num_frequent_sample_indices)  // local
{
  const uint32_t num_frequent_sample_indices = __ldg(d_num_frequent_sample_indices);
  for (uint32_t i = blockIdx.x; i < num_frequent_sample_indices; i += gridDim.x) {
    uint32_t index = frequent_sample_indices[i];
    dtype category = samples[index + global_sample_index_base];
    dtype frequent_index = category_frequent_index[category];
    interaction_layer_input[index * embedding_vec_size + threadIdx.x] =
        frequent_embedding_vectors[frequent_index * embedding_vec_size + threadIdx.x];
  }
}

__global__ void reset_relevant_gradients(float* __restrict__ gradients,
                                         const uint32_t* __restrict__ network_cache_indices,
                                         uint32_t embedding_vec_size,
                                         const uint32_t* d_num_network_cache_indices) {
  const uint32_t num_network_cache_indices = __ldg(d_num_network_cache_indices);
  for (uint32_t i = blockIdx.x; i < num_network_cache_indices; i += gridDim.x)
    gradients[network_cache_indices[i] * embedding_vec_size + threadIdx.x] = 0.0f;
}

__global__ void reset_all_gradients(float* __restrict__ gradients, uint32_t embedding_vec_size) {
  gradients[blockIdx.x * embedding_vec_size + threadIdx.x] = 0.0f;
}

template <typename dtype, typename emtype>
__global__ void frequent_local_reduce(const emtype* __restrict__ gradients_in,
                                      float* __restrict__ gradients_out,
                                      const uint32_t* __restrict__ frequent_sample_indices,
                                      const dtype* __restrict__ local_samples,
                                      const dtype* __restrict__ category_frequent_index,
                                      uint32_t embedding_vec_size,
                                      const uint32_t* d_num_frequent_sample_indices) {
  const uint32_t num_frequent_sample_indices = __ldg(d_num_frequent_sample_indices);

  for (uint32_t i = blockIdx.x; i < num_frequent_sample_indices; i += gridDim.x) {
    uint32_t local_sample_index = frequent_sample_indices[i];
    dtype category = local_samples[local_sample_index];
    dtype frequent_index = category_frequent_index[category];

    atomicAdd(gradients_out + frequent_index * embedding_vec_size + threadIdx.x,
              TypeConvertFunc<float, emtype>::convert(
                  gradients_in[local_sample_index * embedding_vec_size + threadIdx.x]));
  }
}

__global__ void update_model_direct(const float* __restrict__ gradients,
                                    float* const* __restrict__ embedding_vectors_pointers,
                                    const uint32_t* __restrict__ network_cache_indices,
                                    const uint32_t* __restrict__ network_cache_indices_offsets,
                                    uint32_t num_instances, uint32_t network_id,
                                    uint32_t num_frequent_per_model, uint32_t embedding_vec_size,
                                    float lr) {
  const uint32_t offset = __ldg(network_cache_indices_offsets + network_id + 1);
  const uint32_t num_network_cache_indices = __ldg(network_cache_indices_offsets + num_instances);

  for (uint32_t i = blockIdx.x; i < num_network_cache_indices; i += gridDim.x) {
    int vid = (i + offset) % num_network_cache_indices;

    uint32_t frequent_index = network_cache_indices[vid];
    uint32_t model_id = frequent_index / num_frequent_per_model;

    float* embedding_vectors = embedding_vectors_pointers[model_id];

    uint32_t cache_location = frequent_index * embedding_vec_size + threadIdx.x;
    atomicAdd(embedding_vectors + cache_location, -lr * gradients[cache_location]);
  }
}

template <typename dtype>
__global__ void model_cache_indices_mask(const dtype* __restrict__ samples,
                                         const dtype* __restrict__ category_frequent_index,
                                         bool* __restrict__ mask, uint32_t samples_size,
                                         uint32_t local_samples_size, uint32_t num_frequent,
                                         uint32_t num_frequent_per_model, uint32_t model_id) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < samples_size) {
    dtype category = __ldg(samples + tid);
    dtype frequent_index = __ldg(category_frequent_index + category);

    if (frequent_index < num_frequent && frequent_index / num_frequent_per_model == model_id)
      mask[(tid / local_samples_size) * num_frequent_per_model +
           frequent_index % num_frequent_per_model] = true;
  }
}

__global__ void mask_indices_to_buffer_indices(
    uint32_t* __restrict__ model_cache_indices,
    const uint32_t* __restrict__ model_cache_indices_offsets, uint32_t num_instances,
    uint32_t num_frequent_per_model, uint32_t model_id) {
  const uint32_t num_selected = __ldg(model_cache_indices_offsets + num_instances);

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_selected;
       i += blockDim.x * gridDim.x)
    model_cache_indices[i] =
        model_cache_indices[i] % num_frequent_per_model + num_frequent_per_model * model_id;
}

template <typename dtype>
__global__ void network_cache_mask(const dtype* __restrict__ samples,
                                   const dtype* __restrict__ category_frequent_index,
                                   bool* __restrict__ mask, uint32_t offset,
                                   uint32_t local_samples_size, uint32_t num_frequent) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < local_samples_size) {
    dtype category = __ldg(samples + offset + tid);
    dtype frequent_index = __ldg(category_frequent_index + category);

    if (frequent_index < num_frequent) mask[frequent_index] = true;
  }
}

}  // namespace frequent_embedding_kernels

template <typename dtype, typename emtype>
FrequentEmbedding<dtype, emtype>::FrequentEmbedding(const Data<dtype>& data_train,
                                                    const Data<dtype>& data_evaluate,
                                                    const Model<dtype>& model,
                                                    const GPUResource& gpu_resource,
                                                    BuffPtr<emtype>& grouped_wgrad_buff,
                                                    uint32_t embedding_vec_size,
                                                    size_t max_num_frequent_categories)
    : model_(model),
      data_train_(data_train),
      data_evaluate_(data_evaluate),
      data_(data_train),  // Temporary
      gpu_resource(gpu_resource),
      grouped_wgrad_buff_(grouped_wgrad_buff),
      embedding_vec_size_(embedding_vec_size),
      max_num_frequent_categories_(max_num_frequent_categories) {
  size_t universe_batch_size = std::max(data_train.batch_size, data_evaluate.batch_size);
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &frequent_embedding_vectors_);
  if (sizeof(emtype) != sizeof(float)) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &float_frequent_gradients_);
  }

  auto& gradients = get_gradients();
  if (grouped_wgrad_buff == NULL) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &gradients);
  }
  else {
    grouped_wgrad_buff->reserve({max_num_frequent_categories, embedding_vec_size_}, &gradients);
  }

  buf->reserve({max_num_frequent_categories, 1}, &model_cache_indices_);
  buf->reserve({model.num_instances + 1, 1}, &model_cache_indices_offsets_);
  buf->reserve({max_num_frequent_categories, 1}, &network_cache_indices_);
  buf->reserve({model.num_instances + 1, 1}, &network_cache_indices_offsets_);
  buf->reserve({max_num_frequent_categories, 1}, &network_cache_mask_);
  buf->reserve(
      {ceildiv<size_t>(universe_batch_size, model.num_instances) * data_train_.table_sizes.size(),
       1},
      &frequent_sample_indices_);
  buf->reserve({1}, &d_num_frequent_sample_indices_);

  /// TODO: only allocate when necessary (single-node) and workaround in tests
  buf->reserve({model.num_instances, 1}, &embedding_vectors_pointers_);

  // Temporary storage
  calculate_frequent_sample_indices_temp_storage_bytes();
  calculate_model_cache_indices_temp_storage_bytes();
  calculate_network_cache_indices_temp_storage_bytes();
  buf->reserve({frequent_sample_indices_temp_storage_bytes, 1},
               &frequent_sample_indices_temp_storage_);
  buf->reserve({model_cache_indices_temp_storage_bytes, 1}, &model_cache_indices_temp_storage_);
  buf->reserve({network_cache_indices_temp_storage_bytes, 1}, &network_cache_indices_temp_storage_);

  buf->allocate();
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::initialize_embedding_vectors(size_t grouped_wgrad_offset_in_bytes) {
  CudaDeviceContext context(gpu_resource.get_device_id());

  const size_t num_tables = data_.table_sizes.size();
  for (size_t model_id = 0; model_id < model_.num_instances; ++model_id) {
    for (size_t embedding = 0; embedding < num_tables; embedding++) {
      float up_bound = sqrt(1.f / data_.table_sizes[embedding]);
      size_t offset =
          embedding_vec_size_ *
          model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding];
      size_t num_elements =
          embedding_vec_size_ *
          (model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding + 1] -
           model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding]);
      UniformGenerator::fill(frequent_embedding_vectors_.get_ptr() + offset, num_elements,
                             -up_bound, up_bound, gpu_resource.get_sm_count(),
                             gpu_resource.get_replica_uniform_curand_generator(),
                             gpu_resource.get_stream());
    }
  }
  if (grouped_wgrad_buff_ != NULL) {
    // update wgrad tensors
    size_t grad_size = model_.num_frequent * embedding_vec_size_;
    if (sizeof(float) != sizeof(emtype)) {
      auto buf = std::make_shared<ExternalManagedBuffer>((char*)grouped_wgrad_buff_->as_tensor().get_ptr() + grouped_wgrad_offset_in_bytes);
      frequent_gradients_ = Tensor2<emtype>({grad_size}, buf);
    } else {
      auto buf = std::make_shared<ExternalManagedBuffer>((char*)grouped_wgrad_buff_->as_tensor().get_ptr() + grouped_wgrad_offset_in_bytes);
      float_frequent_gradients_ = Tensor2<float>({grad_size}, buf);
    }
  }
}

/* Single-node: refresh needed vectors in the cache of each network
 * Note: each network pulls from the models */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_model(cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  const uint32_t network_id = model_.global_instance_id;
  const uint32_t num_frequent_per_model = model_.num_frequent / num_instances;

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  /* Update cache from models */
  PROFILE_RECORD("fre_forward_model.forward_model.start", stream, false);
  frequent_embedding_kernels::forward_model<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      embedding_vectors_pointers_.get_ptr(), frequent_embedding_vectors_.get_ptr(),
      network_cache_indices_.get_ptr(), network_cache_indices_offsets_.get_ptr(), num_instances,
      network_id, num_frequent_per_model, embedding_vec_size_);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_forward_model.forward_model.stop", stream, false);
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_network(emtype* interaction_layer_input,
                                                       cudaStream_t stream) {
  // concatenate the embedding vectors into the buffer for
  // top-mlp input

  uint32_t samples_per_instance = data_.samples.get_num_elements() / model_.num_instances;
  uint32_t global_sample_index_base = model_.global_instance_id * samples_per_instance;

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  frequent_embedding_kernels::forward_network<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      embedding_vec_size_, global_sample_index_base, frequent_sample_indices_.get_ptr(),
      data_.samples.get_ptr(), model_.category_frequent_index.get_ptr(),
      frequent_embedding_vectors_.get_ptr(), interaction_layer_input,
      d_num_frequent_sample_indices_.get_ptr());
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/* Reduce gradients on each network */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::local_reduce(const emtype* gradients, cudaStream_t stream,
                                                    bool reset_all) {
  const uint32_t& num_instances = model_.num_instances;
  const uint32_t& network_id = model_.global_instance_id;
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  if (reset_all) { /* Set to zero all the gradients */
    if (model_.num_frequent > 0) {
      PROFILE_RECORD("fre_local_reduce.reset_all_gradients.start", stream, false);
      frequent_embedding_kernels::
          reset_all_gradients<<<model_.num_frequent, embedding_vec_size_, 0, stream>>>(
              float_frequent_gradients_.get_ptr(), embedding_vec_size_);
      CK_CUDA_THROW_(cudaPeekAtLastError());
      PROFILE_RECORD("fre_local_reduce.reset_all_gradients.stop", stream, false);
    }
  } else { /* Set to zero the gradients of categories that appear in the batch */
    PROFILE_RECORD("fre_local_reduce.reset_relevant_gradients.start", stream, false);
    frequent_embedding_kernels::
        reset_relevant_gradients<<<n_blocks, embedding_vec_size_, 0, stream>>>(
            float_frequent_gradients_.get_ptr(), network_cache_indices_.get_ptr(),
            embedding_vec_size_, network_cache_indices_offsets_.get_ptr() + num_instances);
    CK_CUDA_THROW_(cudaPeekAtLastError());
    PROFILE_RECORD("fre_local_reduce.reset_relevant_gradients.stop", stream, false);
  }

  /* Local reduce */
  frequent_embedding_kernels::frequent_local_reduce<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      gradients, float_frequent_gradients_.get_ptr(), frequent_sample_indices_.get_ptr(),
      data_.samples.get_ptr() + network_id * local_samples_size,
      model_.category_frequent_index.get_ptr(), embedding_vec_size_,
      d_num_frequent_sample_indices_.get_ptr());
  CK_CUDA_THROW_(cudaPeekAtLastError());

  if (sizeof(emtype) != sizeof(float)) {
    convert_array<<<1000, 128, 0, stream>>>(frequent_gradients_.get_ptr(),
                                            float_frequent_gradients_.get_ptr(),
                                            model_.num_frequent * embedding_vec_size_);
    CK_CUDA_THROW_(cudaPeekAtLastError());
  }
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_model(float lr, cudaStream_t stream) {
  sgd_global_update(get_gradients().get_ptr(), frequent_embedding_vectors_.get_ptr(),
                    model_.num_frequent, embedding_vec_size_, lr, stream);
}

/* Update model for single-node: direct write in category "owner"'s table */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_model_direct(float lr, cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;
  const uint32_t& network_id = model_.global_instance_id;
  const uint32_t num_frequent_per_model = model_.num_frequent / num_instances;

  /// TODO: use emtype frequent_gradients_?

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  /* Update models */
  PROFILE_RECORD("fre_update_model_direct.update_model_direct.start", stream, false);
  frequent_embedding_kernels::update_model_direct<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      float_frequent_gradients_.get_ptr(), embedding_vectors_pointers_.get_ptr(),
      network_cache_indices_.get_ptr(), network_cache_indices_offsets_.get_ptr(), num_instances,
      network_id, num_frequent_per_model, embedding_vec_size_, lr);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_update_model_direct.update_model_direct.stop", stream, false);
}

template <typename dtype>
struct FrequentSampleIndicesSelectOp {
  const dtype* samples;
  const dtype* category_frequent_index;
  uint32_t offset;
  dtype num_frequent;
  __host__ __device__ __forceinline__
  FrequentSampleIndicesSelectOp(const dtype* samples, const dtype* category_frequent_index,
                                uint32_t offset, dtype num_frequent)
      : samples(samples),
        category_frequent_index(category_frequent_index),
        offset(offset),
        num_frequent(num_frequent) {}
  __device__ __forceinline__ bool operator()(const uint32_t& idx) const {
    dtype category = __ldg(samples + offset + idx);
    dtype frequent_index = __ldg(category_frequent_index + category);
    return frequent_index < num_frequent;
  }
};

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_frequent_sample_indices_temp_storage_bytes() {
  size_t max_batch_size = std::max(data_train_.batch_size, data_evaluate_.batch_size);

  size_t local_samples_size = (max_batch_size / model_.num_instances) * data_.table_sizes.size();

  cub::CountingInputIterator<uint32_t> counting(0);
  FrequentSampleIndicesSelectOp<dtype> select_op(nullptr, nullptr, 0, 0);
  cub::DeviceSelect::If(nullptr, frequent_sample_indices_temp_storage_bytes, counting,
                        (uint32_t*)nullptr, (uint32_t*)nullptr, local_samples_size, select_op, 0);
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_frequent_sample_indices(cudaStream_t stream) {
  const size_t num_networks = model_.num_instances;
  size_t local_samples_size = (data_.batch_size / num_networks) * data_.table_sizes.size();

  // Select indices of frequent categories appearing in the local MLP batch
  cub::CountingInputIterator<uint32_t> counting(0);
  FrequentSampleIndicesSelectOp<dtype> select_op(
      data_.samples.get_ptr(), model_.category_frequent_index.get_ptr(),
      model_.global_instance_id * local_samples_size, model_.num_frequent);
  cub::DeviceSelect::If(
      reinterpret_cast<void*>(frequent_sample_indices_temp_storage_.get_ptr()),
      frequent_sample_indices_temp_storage_bytes, counting, frequent_sample_indices_.get_ptr(),
      d_num_frequent_sample_indices_.get_ptr(), local_samples_size, select_op, stream);
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_model_cache_indices_temp_storage_bytes() {
  const size_t num_frequent = max_num_frequent_categories_;

  size_t select_bytes = 0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, num_frequent, 0);

  constexpr uint32_t align = 256;
  model_cache_indices_temp_storage_bytes = alignTo<size_t>(num_frequent, align) + select_bytes;
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_model_cache_indices(cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  const size_t samples_size = data_.batch_size * data_.table_sizes.size();
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  // Note: we assume that the number of frequent categories is a
  // multiple of the number of models!
  const size_t num_frequent_per_model = num_frequent / num_instances;

  /**
   * Explanation of the mask:
   * The model owns num_frequent_per_model categories. For each network,
   * we want to know the categories that appear in their local batch and
   * belong to this model. The mask is the concatenation of num_network
   * sections of size num_frequent_per_model.
   * It has a size num_frequent but does not represent all the frequent
   * categories, only num_networks repetitions of the same categories.
   */

  // Temporary storage
  constexpr uint32_t align = 256;
  char* scratch_ptr = model_cache_indices_temp_storage_.get_ptr();
  size_t scratch_offset = 0;
  bool* d_mask = reinterpret_cast<bool*>(scratch_ptr + scratch_offset);
  scratch_offset += alignTo<size_t>(num_frequent, align);
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr + scratch_offset);
  size_t temp_storage_bytes = model_cache_indices_temp_storage_bytes - scratch_offset;

  /* Initialize the mask to false */
  CK_CUDA_THROW_(cudaMemsetAsync(d_mask, 0, num_frequent, stream));

  /* Compute the mask */
  constexpr size_t TPB_mask = 256;
  size_t n_blocks = ceildiv<size_t>(samples_size, TPB_mask);
  frequent_embedding_kernels::model_cache_indices_mask<<<n_blocks, TPB_mask, 0, stream>>>(
      data_.samples.get_ptr(), model_.category_frequent_index.get_ptr(), d_mask, samples_size,
      local_samples_size, num_frequent, num_frequent_per_model, model_.global_instance_id);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  /* Select categories according to the mask */
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, counting, d_mask, model_cache_indices_.get_ptr(),
      model_cache_indices_offsets_.get_ptr() + num_instances, num_frequent, stream);

  /* Compute offsets */
  constexpr size_t TPB_offsets = 256;
  n_blocks = ceildiv<size_t>(num_instances, TPB_offsets);
  offsets_kernel<<<n_blocks, TPB_offsets, 0, stream>>>(model_cache_indices_.get_ptr(),
                                                       model_cache_indices_offsets_.get_ptr(),
                                                       num_instances, num_frequent_per_model);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  /* Convert to buffer indices */

  constexpr size_t TPB_convert = 256;
  n_blocks = gpu_resource.get_sm_count();
  frequent_embedding_kernels::mask_indices_to_buffer_indices<<<n_blocks, TPB_convert, 0, stream>>>(
      model_cache_indices_.get_ptr(), model_cache_indices_offsets_.get_ptr(), num_instances,
      num_frequent_per_model, model_.global_instance_id);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_network_cache_mask(cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  /* Initialize the mask to false */
  PROFILE_RECORD("fre_calculate_network_cache_indices.memset.start", stream, false);
  CK_CUDA_THROW_(cudaMemsetAsync(network_cache_mask_.get_ptr(), 0, num_frequent, stream));
  PROFILE_RECORD("fre_calculate_network_cache_indices.memset.stop", stream, false);

  /* Compute the mask */
  constexpr size_t TPB_mask = 256;
  size_t n_blocks = ceildiv<size_t>(local_samples_size, TPB_mask);
  PROFILE_RECORD("fre_calculate_network_cache_indices.network_cache_mask.start", stream, false);
  frequent_embedding_kernels::network_cache_mask<<<n_blocks, TPB_mask, 0, stream>>>(
      data_.samples.get_ptr(), model_.category_frequent_index.get_ptr(),
      network_cache_mask_.get_ptr(), model_.global_instance_id * local_samples_size,
      local_samples_size, num_frequent);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_calculate_network_cache_indices.network_cache_mask.stop", stream, false);
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_network_cache_indices_temp_storage_bytes() {
  const size_t num_frequent = max_num_frequent_categories_;

  size_t select_bytes = (size_t)0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, num_frequent, 0);

  network_cache_indices_temp_storage_bytes = select_bytes;
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::calculate_network_cache_indices(cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  // Note: we assume that the number of frequent categories is a
  // multiple of the number of models!
  const size_t num_frequent_per_model = num_frequent / num_instances;

  // Temporary storage
  char* scratch_ptr = network_cache_indices_temp_storage_.get_ptr();
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr);
  size_t temp_storage_bytes = network_cache_indices_temp_storage_bytes;

  /* Select categories according to the mask */
  cub::CountingInputIterator<uint32_t> counting(0);
  PROFILE_RECORD("fre_calculate_network_cache_indices.device_select_flagged.start", stream, false);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting,
                             network_cache_mask_.get_ptr(), network_cache_indices_.get_ptr(),
                             network_cache_indices_offsets_.get_ptr() + num_instances, num_frequent,
                             stream);
  PROFILE_RECORD("fre_calculate_network_cache_indices.device_select_flagged.stop", stream, false);

  /* Compute offsets */
  constexpr size_t TPB_offsets = 256;
  size_t n_blocks = ceildiv<size_t>(num_instances, TPB_offsets);
  PROFILE_RECORD("fre_calculate_network_cache_indices.offsets_kernel.start", stream, false);
  offsets_kernel<<<n_blocks, TPB_offsets, 0, stream>>>(network_cache_indices_.get_ptr(),
                                                       network_cache_indices_offsets_.get_ptr(),
                                                       num_instances, num_frequent_per_model);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_calculate_network_cache_indices.offsets_kernel.stop", stream, false);
}

template class FrequentEmbedding<uint32_t, __half>;
template class FrequentEmbedding<uint32_t, float>;
template class FrequentEmbedding<long long, __half>;
template class FrequentEmbedding<long long, float>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
