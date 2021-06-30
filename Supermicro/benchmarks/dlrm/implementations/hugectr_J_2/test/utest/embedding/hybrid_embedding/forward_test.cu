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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/utils.cuh"
#include "hybrid_embedding_cpu.hpp"
#include "test_common.cuh"

/****************** Frequent and infrequent forward network ******************/

template <typename dtype, typename emtype>
class ForwardNetworkTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
  bool single_node;

 public:
  ForwardNetworkTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size, bool single_node,
                     size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed),
        single_node(single_node) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.forward_network();
    if (!single_node) {
      cpu_embedding.calculate_infrequent_model_indices();
      cpu_embedding.forward_a2a_messages();
    }

    /* Tensors for the interaction layer input and messages */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> interaction_layer_input(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &interaction_layer_input[i]);
    }
    std::vector<Tensor2<emtype>> received_messages(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &received_messages[i]);
    }
    buff->allocate();

    /* Frequent and infrequent forward_network */
    this->build_infrequent();
    this->build_frequent();
    std::vector<std::vector<emtype>> h_interaction_layer_input(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.frequent_embedding_vectors[i],
                    this->frequent_embeddings[i].frequent_embedding_vectors_, this->stream);
      upload_tensor(cpu_embedding.infrequent_embedding_vectors[i],
                    this->infrequent_embeddings[i].infrequent_embedding_vectors_, this->stream);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      this->infrequent_embeddings[i].calculate_network_indices(this->stream);
      this->frequent_embeddings[i].calculate_frequent_sample_indices(this->stream);
      this->frequent_embeddings[i].forward_network(interaction_layer_input[i].get_ptr(),
                                                   this->stream);
      if (single_node) {
        this->infrequent_embeddings[i].forward_network_direct(interaction_layer_input[i].get_ptr(),
                                                              this->stream);
      } else {
        upload_tensor(cpu_embedding.forward_received_messages[i], received_messages[i],
                      this->stream);
        this->infrequent_embeddings[i].forward_network(
            received_messages[i].get_ptr(), interaction_layer_input[i].get_ptr(), this->stream);
      }

      download_tensor(h_interaction_layer_input[i], interaction_layer_input[i], this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      ASSERT_TRUE(compare_array(
          local_batch_size * this->config.num_tables * this->config.embedding_vec_size,
          h_interaction_layer_input[i].data(), cpu_embedding.interaction_layer_input[i].data(),
          1e-2));
    }
  }
};

/************** Frequent embedding forward model (single node) **************/

template <typename dtype, typename emtype>
class FrequentForwardModelTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
 public:
  FrequentForwardModelTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                           size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_frequent_network_cache_indices();
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.generate_gradients();
    cpu_embedding.frequent_reduce_gradients();

    /* Tensors for the gradients */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> gradients(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &gradients[i]);
    }
    buff->allocate();

    /* Frequent update_model */
    this->build_frequent();
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.frequent_embedding_vectors[i],
                    this->frequent_embeddings[i].frequent_embedding_vectors_, this->stream);
      upload_tensor(cpu_embedding.gradients[i], gradients[i], this->stream);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].calculate_network_cache_mask(this->stream);
      this->frequent_embeddings[i].calculate_network_cache_indices(this->stream);
      this->frequent_embeddings[i].calculate_frequent_sample_indices(this->stream);
      this->frequent_embeddings[i].local_reduce(gradients[i].get_ptr(), this->stream, false);
      this->frequent_embeddings[i].update_model_direct(this->config.lr, this->stream);
    }

    /* Frequent forward_model */
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].forward_model(this->stream);
    }

    std::vector<std::vector<float>> updated_vectors(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      download_tensor(updated_vectors[i], this->frequent_embeddings[i].frequent_embedding_vectors_,
                      this->stream);
    }

    /* Reference update_model */
    cpu_embedding.frequent_update_single_node();

    /* Reference forward_model */
    cpu_embedding.frequent_forward_model();

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      updated_vectors[i].resize(this->config.num_frequent * this->config.embedding_vec_size);
      EXPECT_THAT(updated_vectors[i],
                  ::testing::Pointwise(::testing::FloatNear(1e-2),
                                       cpu_embedding.frequent_embedding_vectors[i]));
    }
  }
};

/**************************** Test instantiations ****************************/

static const HybridEmbeddingConfig<uint32_t> config_uint32 = {4, 32, 10, 128, 1000, 128, 0.5f};
static const HybridEmbeddingConfig<long long> config_int64 = {4, 32, 10, 128, 1000, 128, 0.5f};
static const HybridEmbeddingConfig<uint32_t> config_uint32_single_node = {1,    8,   10,  128,
                                                                          1000, 128, 0.5f};
static const HybridEmbeddingConfig<long long> config_int64_single_node = {1,    8,   10,  128,
                                                                          1000, 128, 0.5f};

// Edge cases: no frequent, all frequent
static const HybridEmbeddingConfig<uint32_t> config_no_freq = {4, 32, 10, 128, 1000, 0, 0.5f};
static const HybridEmbeddingConfig<uint32_t> config_all_freq = {4, 32, 10, 128, 1000, 1000, 0.5f};
static const HybridEmbeddingConfig<uint32_t> config_no_freq_single_node = {1,    8, 10,  128,
                                                                           1000, 0, 0.5f};
static const HybridEmbeddingConfig<uint32_t> config_all_freq_single_node = {1,    8,    10,  128,
                                                                            1000, 1000, 0.5f};

/* hybrid_embedding_forward_network_test */

TEST(hybrid_embedding_forward_network_test, uint32_half_64) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_half_64) {
  ForwardNetworkTest<long long, __half>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_half_2048) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_half_2048) {
  ForwardNetworkTest<long long, __half>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_64) {
  ForwardNetworkTest<uint32_t, float>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_float_64) {
  ForwardNetworkTest<long long, float>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_2048) {
  ForwardNetworkTest<uint32_t, float>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_float_2048) {
  ForwardNetworkTest<long long, float>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_128_no_freq) {
  ForwardNetworkTest<uint32_t, float>(config_no_freq, 128, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_128_all_freq) {
  ForwardNetworkTest<uint32_t, float>(config_all_freq, 128, false).run();
}

/* hybrid_embedding_forward_network_single_node_test */

TEST(hybrid_embedding_forward_network_single_node_test, uint32_half_64) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_half_64) {
  ForwardNetworkTest<long long, __half>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_half_2048) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_half_2048) {
  ForwardNetworkTest<long long, __half>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_64) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_float_64) {
  ForwardNetworkTest<long long, float>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_2048) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_float_2048) {
  ForwardNetworkTest<long long, float>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_128_no_freq) {
  ForwardNetworkTest<uint32_t, float>(config_no_freq_single_node, 128, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_128_all_freq) {
  ForwardNetworkTest<uint32_t, float>(config_all_freq_single_node, 128, true).run();
}

/* hybrid_embedding_frequent_forward_model_test */

TEST(hybrid_embedding_frequent_forward_model_test, uint32_half_64) {
  FrequentForwardModelTest<uint32_t, __half>(config_uint32_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_half_64) {
  FrequentForwardModelTest<long long, __half>(config_int64_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_half_2048) {
  FrequentForwardModelTest<uint32_t, __half>(config_uint32_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_half_2048) {
  FrequentForwardModelTest<long long, __half>(config_int64_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_64) {
  FrequentForwardModelTest<uint32_t, float>(config_uint32_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_float_64) {
  FrequentForwardModelTest<long long, float>(config_int64_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_2048) {
  FrequentForwardModelTest<uint32_t, float>(config_uint32_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_float_2048) {
  FrequentForwardModelTest<long long, float>(config_int64_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_128_no_freq) {
  FrequentForwardModelTest<uint32_t, float>(config_no_freq_single_node, 128).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_128_all_freq) {
  FrequentForwardModelTest<uint32_t, float>(config_all_freq_single_node, 128).run();
}
