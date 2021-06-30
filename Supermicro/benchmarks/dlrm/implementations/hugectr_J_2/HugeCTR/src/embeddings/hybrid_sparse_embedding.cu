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

#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include <collectives/all_reduce_comm.hpp>
#include <common.hpp>

namespace HugeCTR {
template <typename dtype, typename emtype>
HybridSparseEmbedding<dtype, emtype>::HybridSparseEmbedding(
    const Tensors2<dtype> &train_input_tensors, const Tensors2<dtype> &evaluate_input_tensors,
    const HybridSparseEmbeddingParams<emtype> &embedding_params,
    const std::vector<BuffPtr<emtype>>& grouped_wgrad_buff,
    const std::shared_ptr<ResourceManager> &resource_manager)
    : train_input_tensors_(train_input_tensors),
      evaluate_input_tensors_(evaluate_input_tensors),
      embedding_params_(embedding_params),
      resource_manager_(resource_manager),
      stream_queues(resource_manager->get_local_gpu_count()),
      event_queues(resource_manager->get_local_gpu_count()),
      grouped_wgrad_buff_(grouped_wgrad_buff),
      grouped_all_reduce_(grouped_wgrad_buff[0] != NULL) {
  try {
    // 0. Error check
    if (embedding_params_.train_batch_size < 1 || embedding_params_.evaluate_batch_size < 1 ||
        embedding_params_.slot_num < 1 || embedding_params_.embedding_vec_size < 1) {
      CK_THROW_(Error_t::WrongInput, "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
    }

    if (embedding_params_.embedding_vec_size > 1024) {
      CK_THROW_(Error_t::WrongInput,
                "the embedding_vec_size can not be more than 1024 in embedding layer");
    }

    size_t total_gpu_count = resource_manager_->get_global_gpu_count();
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    if (train_input_tensors.size() != local_gpu_count ||
        evaluate_input_tensors.size() != local_gpu_count) {
      CK_THROW_(Error_t::WrongInput,
                "either train_input_tensors.size() or evaluate_input_tensors.size() isn't "
                "local_gpu_count_");
    }

    MESSAGE_("Using Hybrid Embedding with train batch " + std::to_string(get_batch_size(true)) +
             " and eval batch " + std::to_string(get_batch_size(false)));

    // 1. initialize optimizer
    for (size_t id = 0; id < local_gpu_count; id++) {
      OptParams<emtype> opt_params;
      opt_params.optimizer = embedding_params_.opt_params.optimizer;
      opt_params.lr = embedding_params_.opt_params.lr;
      opt_params.update_type = embedding_params_.opt_params.update_type;
      opt_params.scaler = embedding_params_.opt_params.scaler;
      opt_params_.emplace_back(opt_params);
    }
    // 2. reserve buffers for different tensors
    assert(bufs_.empty());
    CudaDeviceContext context;
    // 2.1. construct data
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);

      data_statistics_.emplace_back(embedding_params_.slot_size_array, get_batch_size(true),
                                    embedding_params_.num_iterations_statistics);
      data_train_.emplace_back(embedding_params_.slot_size_array, get_batch_size(true), 1);
      data_evaluate_.emplace_back(embedding_params_.slot_size_array, get_batch_size(false), 1);
    }

    // 2.2 construct model
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);

      std::vector<uint32_t> num_instances_per_node(resource_manager_->get_num_process(), 0);
      get_num_instances_per_node(num_instances_per_node);
      model_.emplace_back(embedding_params_.communication_type,
                          resource_manager_->get_local_gpu(i)->get_global_id(),
                          num_instances_per_node, get_categories_num());
    }

    // 2.3 construct calibration
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      calibration_.emplace_back(resource_manager_->get_num_process(), embedding_params_.p_dup_max,
                                embedding_params_.max_all_reduce_bandwidth,
                                embedding_params_.max_all_to_all_bandwidth);
    }

    // 2.4 construct Statistics
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      const size_t num_samples_statistics = embedding_params_.num_iterations_statistics *
                                            get_batch_size(true) * embedding_params_.slot_num;
      statistics_.emplace_back((dtype)num_samples_statistics, embedding_params_.slot_num,
                               model_[i].num_instances, get_categories_num());
    }

    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
      bufs_.emplace_back(buf);
      // 2.5. reserve for train output/ evaluate output tensors
      Tensor2<emtype> tensor;
      buf->reserve({get_batch_size_per_gpu(true), get_slot_num(), get_embedding_vec_size()},
                   &tensor);
      train_output_tensors_.emplace_back(tensor);
      buf->reserve({get_batch_size_per_gpu(false), get_slot_num(), get_embedding_vec_size()},
                   &tensor);
      evaluate_output_tensors_.emplace_back(tensor);

      // 2.6 construct frequent embedding
      frequent_embeddings_.emplace_back(data_train_[i], data_evaluate_[i], model_[i],
                                        get_local_gpu(i),
                                        grouped_wgrad_buff_[i],
                                        get_embedding_vec_size(),
                                        embedding_params_.max_num_frequent_categories);

      // 2.7 construct infrequent embedding
      infrequent_embeddings_.emplace_back(data_train_[i], data_evaluate_[i], model_[i],
                                          get_local_gpu(i), get_embedding_vec_size());

      // 2.8 construct communication
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
        size_t max_buf_size = embedding_params_.embedding_vec_size *
                              std::max(get_batch_size(true), get_batch_size(false)) *
                              embedding_params_.slot_num;
        infrequent_forward_comm_buffers_.emplace_back(buf.get(), max_buf_size);
        infrequent_backward_comm_buffers_.emplace_back(buf.get(), max_buf_size);
        infrequent_forward_comms_.emplace_back(std::make_unique<AllToAll_Multi_NCCL<emtype>>(
            infrequent_forward_comm_buffers_.back().send_buffer,
            infrequent_forward_comm_buffers_.back().recv_buffer,
            infrequent_embeddings_.back().get_model_indices_offsets_ptr(),
            infrequent_embeddings_.back().get_network_indices_offsets_ptr(), &get_local_gpu(i),
            embedding_params_.embedding_vec_size));
        infrequent_backward_comms_.emplace_back(std::make_unique<AllToAll_Multi_NCCL<emtype>>(
            infrequent_backward_comm_buffers_.back().send_buffer,
            infrequent_backward_comm_buffers_.back().recv_buffer,
            infrequent_embeddings_.back().get_network_indices_offsets_ptr(),
            infrequent_embeddings_.back().get_model_indices_offsets_ptr(), &get_local_gpu(i),
            embedding_params_.embedding_vec_size));
      }

      // Construct comm buffers
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
        size_t max_buf_size = embedding_params_.embedding_vec_size *
                              std::max(get_batch_size(true), get_batch_size(false)) *
                              embedding_params_.slot_num;
        infrequent_forward_comm_buffers_.emplace_back(buf.get(), max_buf_size);
        infrequent_backward_comm_buffers_.emplace_back(buf.get(), max_buf_size);
        buf->reserve({local_gpu_count}, &infrequent_forward_comm_buffers_.back().send_buffer_ptrs);
        buf->reserve({local_gpu_count}, &infrequent_backward_comm_buffers_.back().send_buffer_ptrs);
      }

      // For global barrier in eval
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1}, &tensor);
        d_barrier_store_.push_back(tensor);
      }
      buf->allocate();
    }

    // Frequent AR comm init
    if ((embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) ||
        (embedding_params_.communication_type == CommunicationType::IB_NVLink)) {
      
      if (!grouped_all_reduce_) {
        // Do your own all-reduce
        auto ar_comm = resource_manager_->get_ar_comm();
        frequent_embedding_handle_ = ar_comm->register_coll();

        // Frequent all reduce comm
        for (uint32_t i = 0; i < local_gpu_count; i++) {
          int cur_device = get_local_gpu(i).get_device_id();
          CudaDeviceContext context(cur_device);
          ar_comm->set_coll_buf(frequent_embedding_handle_, 
              frequent_embeddings_[i].get_gradients().get_ptr(),
              frequent_embeddings_[i].get_gradients().get_size_in_bytes(),
              i);
          frequent_comms_.emplace_back(std::make_unique<AllReduceComm<emtype>>(
                ar_comm, frequent_embedding_handle_, &get_local_gpu(i)));
        }
        ar_comm->register_coll_buf(frequent_embedding_handle_);
      }
    }

    // Init after buffer allocation
    if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
#ifdef ENABLE_MPI
      ib_comm_ = resource_manager_->get_ib_comm();
      comm_stream_.resize(local_gpu_count);

      std::vector<size_t *> h_model_indices_sizes_ptrs(local_gpu_count);
      std::vector<size_t *> h_network_indices_sizes_ptrs(local_gpu_count);
      std::vector<emtype *> h_fwd_send_buffer_ptrs(local_gpu_count);
      std::vector<emtype *> h_bwd_send_buffer_ptrs(local_gpu_count);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        h_model_indices_sizes_ptrs[i] = infrequent_embeddings_[i].model_indices_sizes_.get_ptr();
        h_network_indices_sizes_ptrs[i] =
            infrequent_embeddings_[i].network_indices_sizes_.get_ptr();
        h_fwd_send_buffer_ptrs[i] = infrequent_forward_comm_buffers_[i].send_buffer.get_ptr();
        h_bwd_send_buffer_ptrs[i] = infrequent_backward_comm_buffers_[i].send_buffer.get_ptr();
      }

      // Forward coll init
      auto infrequent_forward_coll_handle  = ib_comm_->register_hier_a2a_v_coll(true);
      auto ar_comm = resource_manager_->get_ar_comm();
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        int cur_device = get_local_gpu(i).get_device_id();
        context.set_device(cur_device);

        // download pointers
        CK_CUDA_THROW_(
            cudaMemcpyAsync(infrequent_embeddings_[i].model_indices_sizes_ptrs_.get_ptr(),
                            h_model_indices_sizes_ptrs.data(), sizeof(size_t *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        CK_CUDA_THROW_(
            cudaMemcpyAsync(infrequent_embeddings_[i].network_indices_sizes_ptrs_.get_ptr(),
                            h_network_indices_sizes_ptrs.data(), sizeof(size_t *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        CK_CUDA_THROW_(
            cudaMemcpyAsync(infrequent_forward_comm_buffers_[i].send_buffer_ptrs.get_ptr(),
                            h_fwd_send_buffer_ptrs.data(), sizeof(emtype *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        CK_CUDA_THROW_(
            cudaMemcpyAsync(infrequent_backward_comm_buffers_[i].send_buffer_ptrs.get_ptr(),
                            h_bwd_send_buffer_ptrs.data(), sizeof(emtype *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        CK_CUDA_THROW_(cudaStreamSynchronize(get_local_gpu(i).get_stream()));

        // Initialize IB comm
        CK_CUDA_THROW_(cudaStreamCreate(&comm_stream_[i]));
        ib_comm_->set_a2a_coll_stream(infrequent_forward_coll_handle, comm_stream_[i], i);

        ib_comm_->set_a2a_coll_buf(
            infrequent_forward_coll_handle,
            infrequent_forward_comm_buffers_[i].send_buffer.get_ptr(),
            infrequent_forward_comm_buffers_[i].send_buffer.get_size_in_bytes(),
            infrequent_forward_comm_buffers_[i].recv_buffer.get_ptr(),
            infrequent_forward_comm_buffers_[i].recv_buffer.get_size_in_bytes(), i);

        infrequent_forward_comms_.emplace_back(std::make_unique<HierAll2Allv_Multi_IB<emtype>>(
            i, infrequent_forward_coll_handle,
            infrequent_embeddings_[i].model_indices_sizes_ptrs_.get_ptr(), &get_local_gpu(i),
            ib_comm_, comm_stream_[i]));
      }
      ib_comm_->register_a2a_coll_buf(infrequent_forward_coll_handle);

      // Backward coll init
      auto infrequent_backward_coll_handle = ib_comm_->register_hier_a2a_v_coll(true);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        int cur_device = get_local_gpu(i).get_device_id();
        context.set_device(cur_device);

        ib_comm_->set_a2a_coll_stream(infrequent_backward_coll_handle, comm_stream_[i], i);
        ib_comm_->set_a2a_coll_buf(
            infrequent_backward_coll_handle,
            infrequent_backward_comm_buffers_[i].send_buffer.get_ptr(),
            infrequent_backward_comm_buffers_[i].send_buffer.get_size_in_bytes(),
            infrequent_backward_comm_buffers_[i].recv_buffer.get_ptr(),
            infrequent_backward_comm_buffers_[i].recv_buffer.get_size_in_bytes(), i);

        infrequent_backward_comms_.emplace_back(std::make_unique<HierAll2Allv_Multi_IB<emtype>>(
            i, infrequent_backward_coll_handle,
            infrequent_embeddings_[i].network_indices_sizes_ptrs_.get_ptr(), &get_local_gpu(i),
            ib_comm_, comm_stream_[i]));
      }
      ib_comm_->register_a2a_coll_buf(infrequent_backward_coll_handle);
#else
      CK_THROW_(Error_t::WrongInput, "MPI is not enabled but trying to use IB_NVLink_Hier");
#endif
    }

    // 2.9 Single-node: copy the embedding table pointers on each device
    if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      std::vector<const float *> h_infrequent_vectors_pointers(local_gpu_count);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        h_infrequent_vectors_pointers[i] =
            infrequent_embeddings_[i].infrequent_embedding_vectors_.get_ptr();
      }
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        CK_CUDA_THROW_(
            cudaMemcpyAsync(infrequent_embeddings_[i].embedding_vectors_pointers_.get_ptr(),
                            h_infrequent_vectors_pointers.data(), local_gpu_count * sizeof(float *),
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));
      }

      std::vector<const float *> h_frequent_vectors_pointers(local_gpu_count);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        h_frequent_vectors_pointers[i] =
            frequent_embeddings_[i].frequent_embedding_vectors_.get_ptr();
      }
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        CK_CUDA_THROW_(
            cudaMemcpyAsync(frequent_embeddings_[i].embedding_vectors_pointers_.get_ptr(),
                            h_frequent_vectors_pointers.data(), local_gpu_count * sizeof(float *),
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));
      }
    }

    for (auto &output : train_output_tensors_) {
      gradient_pointers_.push_back(output.get_ptr());
    }
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      CK_CUDA_THROW_(cudaMemcpyAsync(infrequent_embeddings_[i].gradients_pointers_.get_ptr(),
                                     gradient_pointers_.data(), local_gpu_count * sizeof(emtype *),
                                     cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename dtype, typename emtype>
HybridSparseEmbedding<dtype, emtype>::~HybridSparseEmbedding() {
  destroy_streams();
  destroy_events();
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::init_model(const Tensors2<dtype> &data, size_t& wgrad_offset_in_bytes) {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t id = 0; id < local_gpu_count; ++id) {
    int cur_device = get_local_gpu(id).get_device_id();
    CudaDeviceContext context(cur_device);
    auto stream = get_local_gpu(id).get_stream();
    data_statistics_[id].data_to_unique_categories(data[id], stream);
    model_[id].init_hybrid_model(calibration_[id], statistics_[id], data_statistics_[id], stream);
    frequent_embeddings_[id].initialize_embedding_vectors(wgrad_offset_in_bytes);
    infrequent_embeddings_[id].initialize_embedding_vectors();

    assert(embedding_params_.max_num_frequent_categories >= model_[id].num_frequent);
  }

  MESSAGE_("Initialized hybrid model with " + std::to_string(model_[0].num_frequent) +
           " frequent categories");

  size_t wgrad_size = model_[0].num_frequent*embedding_params_.embedding_vec_size*sizeof(emtype);
  if ((embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) ||
      (embedding_params_.communication_type == CommunicationType::IB_NVLink)) {

    if (!grouped_all_reduce_) {
      // Manage your own all-reduce
      auto ar_comm = resource_manager_->get_ar_comm();
      ar_comm->update_size(frequent_embedding_handle_, wgrad_size);
    }
    else {
      wgrad_offset_in_bytes += wgrad_size;
    }
  }
}

template <typename dtype, typename emtype>
cudaStream_t HybridSparseEmbedding<dtype, emtype>::pop_stream(uint32_t device_id) {
  cudaStream_t stream;
  if (stream_queues[device_id].size() == 0) {
    CK_CUDA_THROW_(cudaStreamCreate(&stream));
  } else {
    stream = stream_queues[device_id].front();
    stream_queues[device_id].pop();
  }
  return stream;
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::push_stream(uint32_t device_id, cudaStream_t stream) {
  stream_queues[device_id].push(stream);
}

template <typename dtype, typename emtype>
cudaEvent_t HybridSparseEmbedding<dtype, emtype>::pop_event(uint32_t device_id) {
  cudaEvent_t event;
  if (event_queues[device_id].size() == 0) {
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  } else {
    event = event_queues[device_id].front();
    event_queues[device_id].pop();
  }
  return event;
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::push_event(uint32_t device_id, cudaEvent_t event) {
  event_queues[device_id].push(event);
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::destroy_streams() {
  for (uint32_t i = 0; i < stream_queues.size(); i++) {
    while (stream_queues[i].size() > 0) {
      cudaStream_t stream = stream_queues[i].front();
      CK_CUDA_THROW_(cudaStreamDestroy(stream));
      stream_queues[i].pop();
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::destroy_events() {
  for (uint32_t i = 0; i < event_queues.size(); i++) {
    while (event_queues[i].size() > 0) {
      cudaEvent_t event = event_queues[i].front();
      CK_CUDA_THROW_(cudaEventDestroy(event));
      event_queues[i].pop();
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::forward(bool is_train) {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

// Index calculations
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    int cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);
    auto stream = get_local_gpu(i).get_stream();

    auto &data = (is_train) ? data_train_[i] : data_evaluate_[i];
    auto &output = (is_train) ? train_output_tensors_[i] : evaluate_output_tensors_[i];
    auto &input = (is_train) ? train_input_tensors_[i] : evaluate_input_tensors_[i];

    frequent_embeddings_[i].data_ = data;
    infrequent_embeddings_[i].data_ = data;
    PROFILE_RECORD("data_to_unique_categories.start", stream, false);
    data.data_to_unique_categories(input, stream);
    PROFILE_RECORD("data_to_unique_categories.stop", stream, false);

    cudaStream_t stream_frequent_sample_indices = pop_stream(i);
    cudaStream_t stream_model_indices = pop_stream(i);
    cudaStream_t stream_network_indices = pop_stream(i);

    cudaEvent_t event_main = pop_event(i);
    cudaEvent_t event_frequent_sample_indices = pop_event(i);
    cudaEvent_t event_model_indices = pop_event(i);
    cudaEvent_t event_network_indices = pop_event(i);

    // The new streams can only start after previous work in the main stream has completed
    CK_CUDA_THROW_(cudaEventRecord(event_main, stream));
    CK_CUDA_THROW_(cudaStreamWaitEvent(stream_frequent_sample_indices, event_main));
    CK_CUDA_THROW_(cudaStreamWaitEvent(stream_model_indices, event_main));
    CK_CUDA_THROW_(cudaStreamWaitEvent(stream_network_indices, event_main));

    PROFILE_RECORD("index_calculation.start", stream, false);
    PROFILE_RECORD("calculate_frequent_sample_indices.start", stream_frequent_sample_indices,
                   false);
    frequent_embeddings_[i].calculate_frequent_sample_indices(stream_frequent_sample_indices);
    PROFILE_RECORD("calculate_frequent_sample_indices.stop", stream_frequent_sample_indices, false,
                   -1, std::string("num_frequent: ") + std::to_string(model_[i].num_frequent));
    CK_CUDA_THROW_(cudaEventRecord(event_frequent_sample_indices, stream_frequent_sample_indices));

    PROFILE_RECORD("inf_calculate_model_indices.start", stream_model_indices, false);
    infrequent_embeddings_[i].calculate_model_indices(stream_model_indices);
    PROFILE_RECORD("inf_calculate_model_indices.stop", stream_model_indices, false);
    CK_CUDA_THROW_(cudaEventRecord(event_model_indices, stream_model_indices));

    PROFILE_RECORD("inf_calculate_network_indices.start", stream_network_indices, false);
    infrequent_embeddings_[i].calculate_network_indices(stream_network_indices);
    PROFILE_RECORD("inf_calculate_network_indices.stop", stream_network_indices, false);
    CK_CUDA_THROW_(cudaEventRecord(event_network_indices, stream_network_indices));

    if ((embedding_params_.communication_type == CommunicationType::IB_NVLink) ||
        (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier)) {
      // Join streams to the main stream
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_frequent_sample_indices));
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_model_indices));
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_network_indices));
      PROFILE_RECORD("index_calculation.stop", stream, false);
      // stream push and pop order is critical to and only to profiler, don't change it.
      push_stream(i, stream_frequent_sample_indices);
      push_stream(i, stream_model_indices);
      push_stream(i, stream_network_indices);

      push_event(i, event_main);
      push_event(i, event_frequent_sample_indices);
      push_event(i, event_model_indices);
      push_event(i, event_network_indices);

    } else {  // Assuming single node
      cudaStream_t stream_network_cache_indices = pop_stream(i);
      cudaEvent_t event_network_cache_indices = pop_event(i);
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream_network_cache_indices, event_main));
      PROFILE_RECORD("single_node_fre_calculate_network_cache_indices.start",
                     stream_network_cache_indices, false);
      frequent_embeddings_[i].calculate_network_cache_mask(stream_network_cache_indices);
      frequent_embeddings_[i].calculate_network_cache_indices(stream_network_cache_indices);
      PROFILE_RECORD("single_node_fre_calculate_network_cache_indices.stop",
                     stream_network_cache_indices, false);
      CK_CUDA_THROW_(cudaEventRecord(event_network_cache_indices, stream_network_cache_indices));

      // Join streams to the main stream
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_frequent_sample_indices));
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_model_indices));
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_network_indices));
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream, event_network_cache_indices));
      PROFILE_RECORD("index_calculation.stop", stream, false);

      // stream push and pop order is critical to and only to profiler, don't change it.
      push_stream(i, stream_frequent_sample_indices);
      push_stream(i, stream_model_indices);
      push_stream(i, stream_network_indices);
      push_stream(i, stream_network_cache_indices);

      push_event(i, event_main);
      push_event(i, event_frequent_sample_indices);
      push_event(i, event_model_indices);
      push_event(i, event_network_indices);
      push_event(i, event_network_cache_indices);
    }
  }

// Forward model/network
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    int cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);
    auto stream = get_local_gpu(i).get_stream();

    auto &data = (is_train) ? data_train_[i] : data_evaluate_[i];
    auto &output = (is_train) ? train_output_tensors_[i] : evaluate_output_tensors_[i];
    auto &input = (is_train) ? train_input_tensors_[i] : evaluate_input_tensors_[i];

    PROFILE_RECORD("hybrid_embedding.forward.start", stream, false);

    if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      CK_CUDA_THROW_(cudaStreamSynchronize(stream));
      PROFILE_RECORD("multi_node_fre_forward_network.start", stream, false);
      frequent_embeddings_[i].forward_network(output.get_ptr(), stream);
      PROFILE_RECORD("multi_node_fre_forward_network.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_forward_model.start", stream, false);
      infrequent_embeddings_[i].forward_model(
          infrequent_forward_comm_buffers_[i].send_buffer.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_forward_model.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_forward_a2a.start", stream, false);
      infrequent_forward_comms_[i]->communicate();
      PROFILE_RECORD("multi_node_inf_forward_a2a.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_forward_network.start", stream, false);
      infrequent_embeddings_[i].forward_network(
          infrequent_forward_comm_buffers_[i].recv_buffer.get_ptr(), output.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_forward_network.stop", stream, false);

    } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      PROFILE_RECORD("multi_node_inf_calculate_model_indices_sizes_from_offsets.start", stream,
                     false);
      infrequent_embeddings_[i].calculate_model_indices_sizes_from_offsets(stream);
      PROFILE_RECORD("multi_node_inf_calculate_model_indices_sizes_from_offsets.stop", stream,
                     false);
      PROFILE_RECORD("multi_node_inf_calculate_network_indices_sizes_from_offsets.start", stream,
                     false);
      infrequent_embeddings_[i].calculate_network_indices_sizes_from_offsets(stream);
      PROFILE_RECORD("multi_node_inf_calculate_network_indices_sizes_from_offsets.stop", stream,
                     false);
      infrequent_forward_comms_[i]->update_sizes();
      PROFILE_RECORD("multi_node_fre_forward_network.start", stream, false);
      frequent_embeddings_[i].forward_network(output.get_ptr(), stream);
      PROFILE_RECORD("multi_node_fre_forward_network.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_fused_intra_forward_model.start", stream, false);
      infrequent_embeddings_[i].fused_intra_forward_model(
          infrequent_forward_comm_buffers_[i].send_buffer_ptrs.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_fused_intra_forward_model.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_forward_a2a.start", stream, false);
      infrequent_forward_comms_[i]->communicate();
      PROFILE_RECORD("multi_node_inf_forward_a2a.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_hier_forward_network.start", stream, false);
      infrequent_embeddings_[i].hier_forward_network(
          infrequent_forward_comm_buffers_[i].recv_buffer.get_ptr(), output.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_hier_forward_network.stop", stream, false);
      infrequent_backward_comms_[i]->update_sizes();
      if (!is_train) { 
        // Global barrier
        CK_NCCL_THROW_(ncclAllReduce(
              (const void*)d_barrier_store_[i].get_ptr(),
              d_barrier_store_[i].get_ptr(),
              sizeof(uint32_t), NcclDataType<uint32_t>::getType(), ncclSum,
              get_local_gpu(i).get_nccl(), stream));
      }
    }
    else {  // Assuming single node

      PROFILE_RECORD("single_node_fre_forward_model.start", stream, false);
      frequent_embeddings_[i].forward_model(stream);
      PROFILE_RECORD("single_node_fre_forward_model.stop", stream, false);

      PROFILE_RECORD("single_node_fre_forward_network.start", stream, false);
      frequent_embeddings_[i].forward_network(output.get_ptr(), stream);
      PROFILE_RECORD("single_node_fre_forward_network.stop", stream, false);

      PROFILE_RECORD("single_node_inf_forward_network_direct.start", stream, false);
      infrequent_embeddings_[i].forward_network_direct(output.get_ptr(), stream);
      PROFILE_RECORD("single_node_inf_forward_network_direct.stop", stream, false);
    }
    PROFILE_RECORD("hybrid_embedding.forward.stop", stream, false);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    int cur_device = get_local_gpu(i).get_device_id();
    auto stream = get_local_gpu(i).get_stream();
    CudaDeviceContext context(cur_device);

    PROFILE_RECORD("hybrid_embedding.backward.start", stream, false);
    PROFILE_RECORD("fre_local_reduce.start", stream, false);
    bool reset_all = ((embedding_params_.communication_type == CommunicationType::IB_NVLink) ||
                      (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier));
    frequent_embeddings_[i].local_reduce(train_output_tensors_[i].get_ptr(), stream, reset_all);
    PROFILE_RECORD("fre_local_reduce.stop", stream, false);

    if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      PROFILE_RECORD("multi_node_fre_backward_allreduce.start", stream, false);
      if (!grouped_all_reduce_) { frequent_comms_[i]->communicate(); }
      PROFILE_RECORD("multi_node_fre_backward_allreduce.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_update_network.start", stream, false);
      infrequent_embeddings_[i].update_network(
          train_output_tensors_[i].get_ptr(),
          infrequent_backward_comm_buffers_[i].send_buffer.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_update_network.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_backward_a2a.start", stream, false);
      infrequent_backward_comms_[i]->communicate();
      PROFILE_RECORD("multi_node_inf_backward_a2a.stop", stream, false);
    } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      PROFILE_RECORD("multi_node_fre_backward_allreduce.start", stream, false);
      if (!grouped_all_reduce_) { frequent_comms_[i]->communicate(); }
      PROFILE_RECORD("multi_node_fre_backward_allreduce.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_fused_intra_update_network.start", stream, false);
      infrequent_embeddings_[i].fused_intra_update_network(
          train_output_tensors_[i].get_ptr(),
          infrequent_backward_comm_buffers_[i].send_buffer_ptrs.get_ptr(), stream);
      PROFILE_RECORD("multi_node_inf_fused_intra_update_network.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_backward_a2a.start", stream, false);
      infrequent_backward_comms_[i]->communicate();
      PROFILE_RECORD("multi_node_inf_backward_a2a.stop", stream, false);
    }
    PROFILE_RECORD("hybrid_embedding.backward.stop", stream, false);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::update_params() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    int cur_device = get_local_gpu(i).get_device_id();
    auto stream = get_local_gpu(i).get_stream();
    CudaDeviceContext context(cur_device);
    float lr_scale = opt_params_[i].lr / opt_params_[i].scaler;

    PROFILE_RECORD("hybrid_embedding.update_params.start", stream, false);
    if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      PROFILE_RECORD("multi_node_fre_update_model.start", stream, false);
      frequent_embeddings_[i].update_model(lr_scale, stream);
      PROFILE_RECORD("multi_node_fre_update_model.stop", stream, false);

      PROFILE_RECORD("multi_node_inf_update_model.start", stream, false);
      infrequent_embeddings_[i].update_model(
          infrequent_backward_comm_buffers_[i].recv_buffer.get_ptr(), lr_scale, stream);
      PROFILE_RECORD("multi_node_inf_update_model.stop", stream, false);
    } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      PROFILE_RECORD("multi_node_fre_update_model.start", stream, false);
      frequent_embeddings_[i].update_model(lr_scale, stream);
      PROFILE_RECORD("multi_node_fre_update_model.stop", stream, false);
      PROFILE_RECORD("multi_node_inf_hier_update_model.start", stream, false);
      infrequent_embeddings_[i].hier_update_model(
          infrequent_backward_comm_buffers_[i].recv_buffer.get_ptr(), lr_scale, stream);
      PROFILE_RECORD("multi_node_inf_hier_update_model.stop", stream, false);
    } else {  // Assuming single node
      PROFILE_RECORD("single_node_fre_update_model_direct.start", stream, false);
      frequent_embeddings_[i].update_model_direct(lr_scale, stream);
      PROFILE_RECORD("single_node_fre_update_model_direct.stop", stream, false);

      PROFILE_RECORD("single_node_inf_update_model_direct.start", stream, false);
      infrequent_embeddings_[i].update_model_direct(lr_scale, stream);
      PROFILE_RECORD("single_node_inf_update_model_direct.stop", stream, false);
    }
    PROFILE_RECORD("hybrid_embedding.update_params.stop", stream, false);

#ifdef ENABLE_PROFILING
    bool should_run = PROFILE_RECORD_DATA("hybrid_run_time_params.start", stream);
    if (should_run) {
      std::string general_info =
          std::string("{\"global_batch_size\":") +
          std::to_string(embedding_params_.train_batch_size) + std::string(",") +
          std::string("\"slots_num\":") + std::to_string(embedding_params_.slot_num) +
          std::string(",") + std::string("\"total_gpu_count\":") +
          std::to_string(resource_manager_->get_global_gpu_count()) + std::string(",") +
          std::string("\"local_gpu_count\":") +
          std::to_string(resource_manager_->get_local_gpu_count()) + std::string(",") +
          std::string("\"total_categories\":") + std::to_string(model_[0].num_categories) +
          std::string(",") + std::string("\"bytes_of_dtype\":") + std::to_string(sizeof(dtype)) +
          std::string(",") + std::string("\"bytes_of_emtype\":") + std::to_string(sizeof(emtype)) +
          std::string(",") + std::string("\"embedding_vec_size\":") +
          std::to_string(embedding_params_.embedding_vec_size) + std::string(",");
      std::vector<uint32_t> num_frequent_categories;
      download_tensor(num_frequent_categories,
                      frequent_embeddings_[i].d_num_frequent_sample_indices_, stream);
      std::vector<uint32_t> infrequent_model_indices_offset;
      download_tensor(infrequent_model_indices_offset,
                      infrequent_embeddings_[i].model_indices_offsets_, stream);
      std::vector<uint32_t> infrequent_network_indices_offset;
      download_tensor(infrequent_network_indices_offset,
                      infrequent_embeddings_[i].network_indices_offsets_, stream);
      std::vector<uint32_t> network_cache_indices_offsets_;
      download_tensor(network_cache_indices_offsets_,
                      frequent_embeddings_[i].network_cache_indices_offsets_, stream);
      std::string device_info = std::string("\"num_frequent\":") +
                                std::to_string(model_[i].num_frequent) + std::string(",");
      device_info =
          device_info + std::string("\"num_infrequent\":") +
          std::to_string(model_[i].h_infrequent_model_table_offsets[embedding_params_.slot_num]) +
          std::string(",");
      device_info = device_info + std::string("\"num_frequent_samples\":") +
                    std::to_string(num_frequent_categories[0]) + std::string(",");
      device_info = device_info + std::string("\"infrequent_model_indices_offset\": [");
      for (auto size : infrequent_model_indices_offset) {
        device_info = device_info + std::to_string(size) + std::string(",");
      }
      device_info.pop_back();
      device_info = device_info + std::string("],");

      device_info = device_info + std::string("\"infrequent_network_indices_offset\": [");
      for (auto size : infrequent_network_indices_offset) {
        device_info = device_info + std::to_string(size) + std::string(",");
      }
      device_info.pop_back();
      device_info = device_info + std::string("],");

      device_info = device_info + std::string("\"network_cache_indices_offsets_\": [");
      for (auto size : network_cache_indices_offsets_) {
        device_info = device_info + std::to_string(size) + std::string(",");
      }
      device_info.pop_back();
      device_info = device_info + std::string("]");
      std::string run_time_info = general_info + device_info + std::string("}");

      PROFILE_RECORD_DATA("hybrid_run_time_params.stop", stream, run_time_info);
    }
#endif
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::init_params() {
  // TODO: create init_params()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::load_parameters(std::istream &stream) {
  // TODO: create load_parameters()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::dump_parameters(std::ostream &stream) const {
  // TODO: create dump_parameters()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::set_learning_rate(float lr) {
  for (auto &param : opt_params_) {
    param.lr = lr;
  }
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_params_num() const {
  return 0;
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_vocabulary_size() const {
  // TODO: create get_vocabulary_size()
  return 0;
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_max_vocabulary_size() const {
  // TODO: create get_max_vocabulary_size()
  return 0;
}

template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_train_output_tensors() const {
  return tensors_to_bags(train_output_tensors_);
}

template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_evaluate_output_tensors() const {
  return tensors_to_bags(evaluate_output_tensors_);
}

template class HybridSparseEmbedding<uint32_t, __half>;
template class HybridSparseEmbedding<uint32_t, float>;
template class HybridSparseEmbedding<long long, __half>;
template class HybridSparseEmbedding<long long, float>;
}  // namespace HugeCTR
