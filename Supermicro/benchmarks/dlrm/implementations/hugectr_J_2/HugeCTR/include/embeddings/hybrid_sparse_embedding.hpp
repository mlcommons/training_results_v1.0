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

#pragma once

#include <gpu_barrier.hpp>
#include <collectives/ib_comm.hpp>
#include <queue>
#include <random>
#include <resource_manager.hpp>
#include <utils.hpp>
#include <vector>

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include <collectives/ib_comm.hpp>
#include <collectives/all_reduce_comm.hpp>
#include <gpu_barrier.hpp>

using namespace HugeCTR::hybrid_embedding;

namespace HugeCTR {

///
/// Interface class for the hybrid embedding to HugeCTR. It is responsible for
/// persistent gpu memory allocation.
///
template <typename dtype, typename emtype>
class HybridSparseEmbedding : public IEmbedding {
 private:
  // Embedding models, one instance per frequent and the infrequent embedding
  // for each mlp-network in the train session.
  //

  // data-parallel embedding model
  std::vector<FrequentEmbedding<dtype, emtype>> frequent_embeddings_;
  // model-parallel embedding model
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings_;
  // performs the communication scheme
  std::vector<std::unique_ptr<Communication>> frequent_comms_, infrequent_forward_comms_,
      infrequent_backward_comms_;
  std::vector<AllToAllStorage<emtype>> infrequent_forward_comm_buffers_,
      infrequent_backward_comm_buffers_;
  std::vector<const emtype *> gradient_pointers_;

  // Hier A2Av / custom AR impl
#ifdef ENABLE_MPI
  std::vector<cudaStream_t> comm_stream_;
  IbComm* ib_comm_;
  AllReduceInPlaceComm::Handle barrier_handle_;
#endif
  
  AllReduceInPlaceComm::Handle frequent_embedding_handle_;
  Tensors2<uint32_t> d_barrier_store_;

  // model_, data_, calibration_ and statistics_ are replications of the model
  // and input data on each gpu. The HybridSparseEmbedding class manages
  // it's scope / frees the memory.
  std::vector<Model<dtype>> model_;
  std::vector<Data<dtype>> data_train_;
  std::vector<Data<dtype>> data_evaluate_;
  std::vector<Data<dtype>> data_statistics_;
  std::vector<CalibrationData> calibration_;
  std::vector<Statistics<dtype>> statistics_;

  // added by kefeng
  // std::vector<CudaPreAllocator> pre_alloc_bufs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs_;

  std::vector<std::queue<cudaStream_t>> stream_queues;
  std::vector<std::queue<cudaEvent_t>> event_queues;

  Tensors2<dtype> train_input_tensors_;
  Tensors2<dtype> evaluate_input_tensors_;
  HybridSparseEmbeddingParams<emtype> embedding_params_;
  std::shared_ptr<ResourceManager> resource_manager_;

  Tensors2<emtype> train_output_tensors_;    /**< The output tensors. */
  Tensors2<emtype> evaluate_output_tensors_; /**< The output tensors. */
  template <typename T> 
  using BuffPtr = std::shared_ptr<BufferBlock2<T>>;
  std::vector<BuffPtr<emtype>> grouped_wgrad_buff_;
  bool grouped_all_reduce_ = false;

  std::vector<OptParams<emtype>> opt_params_; /**< Optimizer params. */

 protected:
  size_t get_batch_size(bool is_train) const {
    if (is_train) {
      return embedding_params_.train_batch_size;
    } else {
      return embedding_params_.evaluate_batch_size;
    }
  }
  size_t get_universal_batch_size() const {
    return std::max(embedding_params_.train_batch_size, embedding_params_.evaluate_batch_size);
  }
  size_t get_batch_size_per_gpu(bool is_train) const {
    return get_batch_size(is_train) / resource_manager_->get_global_gpu_count();
  }
  size_t get_embedding_vec_size() const { return embedding_params_.embedding_vec_size; }
  size_t get_slot_num() const { return embedding_params_.slot_num; }
  void get_num_instances_per_node(std::vector<uint32_t> &num_instances_per_node) {
    uint32_t total_gpu_count = resource_manager_->get_global_gpu_count();
    for (uint32_t gid = 0; gid < total_gpu_count; ++gid) {
      uint32_t nodeid = resource_manager_->get_process_id_from_gpu_global_id(gid);
      num_instances_per_node[nodeid] = num_instances_per_node[nodeid] + 1;
    }
    return;
  }

  const GPUResource &get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  size_t get_categories_num() {
    size_t num_categories = 0;
    for (size_t i = 0; i < embedding_params_.slot_size_array.size(); ++i) {
      num_categories += embedding_params_.slot_size_array[i];
    }
    return num_categories;
  }

  cudaStream_t pop_stream(uint32_t device_id);
  void push_stream(uint32_t device_id, cudaStream_t stream);
  cudaEvent_t pop_event(uint32_t device_id);
  void push_event(uint32_t device_id, cudaEvent_t event);
  void destroy_streams();
  void destroy_events();

 public:
  void load_parameters(const TensorBag2 &keys, const Tensor2<float> &embeddings,
                       size_t num) override {}
  void dump_parameters(TensorBag2 keys, Tensor2<float> &embeddings, size_t *num) const override {}
  void reset() override {}
  void check_overflow() const override {}
  void get_forward_results_tf(const bool is_train, const bool on_gpu,
                              void *const forward_result) override {}
  cudaError_t update_top_gradients(const bool on_gpu, const void *const top_gradients) override { throw; }

  HybridSparseEmbedding(const Tensors2<dtype> &train_input_tensors,
                        const Tensors2<dtype> &evaluate_input_tensors,
                        const HybridSparseEmbeddingParams<emtype> &embedding_params,
                        const std::vector<BuffPtr<emtype>>& grouped_wgrad_buff,
                        const std::shared_ptr<ResourceManager> &resource_manager);
  ~HybridSparseEmbedding();

  void init_model(const Tensors2<dtype> &data, size_t& wgrad_offset);

  void forward(bool is_train) override;
  void backward() override;
  void update_params() override;
  void init_params() override;
  void load_parameters(std::istream &stream) override;
  void dump_parameters(std::ostream &stream) const override;
  void set_learning_rate(float lr) override;
  size_t get_params_num() const override;
  size_t get_vocabulary_size() const override;
  size_t get_max_vocabulary_size() const override;
  std::vector<TensorBag2> get_train_output_tensors() const override;
  std::vector<TensorBag2> get_evaluate_output_tensors() const override;

  // just a simulate function
  void gen_mock_data(Tensors2<dtype> &mock_data_v, cudaStream_t stream) {
    std::mt19937 rnd(12345);
    size_t num_samples = embedding_params_.num_iterations_statistics * get_batch_size(true);
    std::vector<dtype> h_mock_data(num_samples * get_slot_num());
    for (size_t sample = 0; sample < num_samples; ++sample) {
      for (size_t slot = 0; slot < get_slot_num(); ++slot) {
        size_t slot_size = embedding_params_.slot_size_array[slot];
        h_mock_data[sample * get_slot_num() + slot] = rnd() % slot_size;
      }
    }

    for (size_t id = 0; id < resource_manager_->get_local_gpu_count(); ++id) {
      int cur_device = get_local_gpu(id).get_device_id();
      CudaDeviceContext context;
      context.set_device(cur_device);
      std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
      Tensor2<dtype> mock_data;
      buf->reserve({num_samples * get_slot_num(), 1}, &mock_data);
      buf->allocate();
      upload_tensor<dtype>(h_mock_data, mock_data, stream);
      mock_data_v.push_back(mock_data);
    }
  }

};

}  // namespace HugeCTR
