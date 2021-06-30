/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <collectives/all_reduce_comm.hpp>
#include <collectives/ib_comm.hpp>
#include <utils.hpp>

namespace HugeCTR
{

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create_nccl(
    size_t num_process, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources)
  {
    if (use_mixed_precision) {
      return std::make_shared<NCCLARInplaceComm<__half>>(num_process, gpu_resources);
    } else {
      return std::make_shared<NCCLARInplaceComm<float>>(num_process, gpu_resources);
    }
  }
      
#ifdef ENABLE_MPI

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create_oneshot(
    size_t num_process, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm)
  {
    if (num_process == 1) {
      CK_THROW_(Error_t::WrongInput, "Oneshot algo is not defined for single node");
    }
    if (use_mixed_precision) {
      return std::make_shared<OneshotMultiARInplaceComm<__half>>(ib_comm, num_process, gpu_resources);
    } else {
      return std::make_shared<OneshotMultiARInplaceComm<float>>(ib_comm, num_process, gpu_resources);
    }
  }

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create(
    size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm)
  {
    return (algo == AllReduceAlgo::ONESHOT) ? 
      create_oneshot(num_process, use_mixed_precision, gpu_resources, ib_comm) :
      create_nccl(num_process, use_mixed_precision, gpu_resources);
  }

#else
  
  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create(
    size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources)
  {
    if (algo == AllReduceAlgo::ONESHOT) {
      CK_THROW_(Error_t::WrongInput, "Oneshot algo can't be used without MPI");
    }
    return create_nccl(num_process, use_mixed_precision, gpu_resources);
  }

#endif

#ifdef ENABLE_MPI
  template <typename T>
  OneshotMultiARInplaceComm<T>::OneshotMultiARInplaceComm(
      IbComm* ib_comm,
      size_t num_procs,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources):
    ib_comm_(ib_comm),
    num_procs_(num_procs),
    gpu_resources_(gpu_resources),
    num_gpus_(gpu_resources.size())
  {
  }

  template <typename T>
  AllReduceInPlaceComm::Handle OneshotMultiARInplaceComm<T>::register_coll()
  {
    ar_ctx_.emplace_back(std::make_unique<ARContext>());
    Handle handle = (Handle)(ar_ctx_.size() - 1);
    auto& ar_ctx_g = ar_ctx_[handle];
    ar_ctx_g->ctx_.resize(num_gpus_);
    ar_ctx_[handle]->ib_comm_handle_ = ib_comm_->register_ar_coll();

    return handle;
  }

  template<typename T>
  void OneshotMultiARInplaceComm<T>::set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    ctx_g.ar_ptr_ = ar_ptr;
    if ((ctx->ar_size_ != 0) && (ctx->ar_size_ != ar_size)) {
      CK_THROW_(Error_t::WrongInput, "AR size mismatch");
    }
    ctx->ar_size_ = ar_size;
    ib_comm_->set_ar_coll_buf<T>(ctx->ib_comm_handle_, ar_ptr, ar_size, g);
    // MESSAGE_("Oneshot AR size: " + std::to_string(ar_size));
  }

  template<typename T>
  void OneshotMultiARInplaceComm<T>::update_size(Handle coll, const size_t ar_size)
  {
    auto& ctx = ar_ctx_[coll];
    ctx->ar_size_ = ar_size;
    ib_comm_->update_size(ctx->ib_comm_handle_, ar_size);
    // MESSAGE_("Oneshot AR size updated to: " + std::to_string(ar_size));
  }

  template <typename T>
  void OneshotMultiARInplaceComm<T>::register_coll_buf(Handle coll)
  {
    auto& ctx = ar_ctx_[coll];
    ib_comm_->register_ar_coll_buf(ctx->ib_comm_handle_);
  }

  template <typename T>
  void OneshotMultiARInplaceComm<T>::all_reduce(AllReduceInPlaceComm::Handle coll, cudaStream_t stream, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    ib_comm_->all_reduce<T>(ctx->ib_comm_handle_, stream, g);
  }
  
  template class OneshotMultiARInplaceComm<__half>;
  template class OneshotMultiARInplaceComm<float>;
#endif

  template <typename T>
  NCCLARInplaceComm<T>::NCCLARInplaceComm(
      size_t num_procs,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources):
    num_procs_(num_procs),
    gpu_resources_(gpu_resources),
    num_gpus_(gpu_resources.size())
  {
  }

  template <typename T>
  AllReduceInPlaceComm::Handle NCCLARInplaceComm<T>::register_coll()
  {
    ar_ctx_.emplace_back(std::make_unique<ARContext>());
    Handle handle = (Handle)(ar_ctx_.size() - 1);
    auto& ar_ctx_g = ar_ctx_[handle];
    ar_ctx_g->ctx_.resize(num_gpus_);
    
    return handle;
  }

  template<typename T>
  void NCCLARInplaceComm<T>::set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    ctx_g.ar_ptr_ = ar_ptr;
    if ((ctx->ar_size_ != 0) && (ctx->ar_size_ != ar_size)) {
      CK_THROW_(Error_t::WrongInput, "AR size mismatch");
    }
    ctx->ar_size_ = ar_size;
    // MESSAGE_("NCCL AR size: " + std::to_string(ar_size));
  }

  template<typename T>
  void NCCLARInplaceComm<T>::update_size(Handle coll, const size_t ar_size)
  {
    auto& ctx = ar_ctx_[coll];
    ctx->ar_size_ = ar_size;
    // MESSAGE_("NCCL AR size updated to: " + std::to_string(ar_size));
  }

  template<typename T>
  void NCCLARInplaceComm<T>::register_coll_buf(Handle coll)
  {
  }

  template <typename T>
  void NCCLARInplaceComm<T>::all_reduce(AllReduceInPlaceComm::Handle coll, cudaStream_t stream, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    CK_NCCL_THROW_(ncclAllReduce(
          (const void*) ctx_g.ar_ptr_, ctx_g.ar_ptr_,
          ctx->ar_size_ / sizeof(T),
          NcclDataType<T>::getType(),
          ncclSum,
          gpu_resources_[g]->get_nccl(),
          stream));
  }

  template class NCCLARInplaceComm<__half>;
  template class NCCLARInplaceComm<float>;
}
