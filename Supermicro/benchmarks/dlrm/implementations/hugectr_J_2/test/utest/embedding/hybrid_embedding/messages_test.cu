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

/**************** Infrequent embedding: forward sent message ****************/

template <typename dtype, typename emtype>
class ForwardSentMessageTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  ForwardSentMessageTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                         size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_infrequent_model_indices();
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.forward_a2a_messages();

    /* Tensors and vectors for the generated messages */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> sent_messages(this->num_instances);
    std::vector<std::vector<emtype>> h_sent_messages(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({this->batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &sent_messages[i]);
    }
    buff->allocate();

    /* Infrequent forward_model */
    this->build_infrequent();
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.infrequent_embedding_vectors[i],
                    this->infrequent_embeddings[i].infrequent_embedding_vectors_, this->stream);
      this->infrequent_embeddings[i].calculate_model_indices(this->stream);
      this->infrequent_embeddings[i].forward_model(sent_messages[i].get_ptr(), this->stream);
      download_tensor(h_sent_messages[i], sent_messages[i], this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      uint32_t message_size = this->config.embedding_vec_size *
                              cpu_embedding.model_indices_offsets[i][this->num_instances];
      ASSERT_TRUE(compare_array(message_size, h_sent_messages[i].data(),
                                cpu_embedding.forward_sent_messages[i].data(), 1e-2));
    }
  }
};

/**************** Infrequent embedding: backward sent message ****************/

template <typename dtype, typename emtype>
class BackwardSentMessageTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  BackwardSentMessageTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                          size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_infrequent_model_indices();
    cpu_embedding.calculate_infrequent_network_indices();
    cpu_embedding.generate_gradients();
    cpu_embedding.backward_a2a_messages();

    /* Tensors and vectors for the gradients and generated messages */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> sent_messages(this->num_instances);
    std::vector<std::vector<emtype>> h_sent_messages(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &sent_messages[i]);
    }
    std::vector<Tensor2<emtype>> gradients(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &gradients[i]);
    }
    buff->allocate();

    /* Infrequent update_network */
    this->build_infrequent();
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.gradients[i], gradients[i], this->stream);
      this->infrequent_embeddings[i].calculate_network_indices(this->stream);
      this->infrequent_embeddings[i].update_network(gradients[i].get_ptr(),
                                                    sent_messages[i].get_ptr(), this->stream);
      download_tensor(h_sent_messages[i], sent_messages[i], this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      uint32_t message_size = this->config.embedding_vec_size *
                              cpu_embedding.network_indices_offsets[i][this->num_instances];
      ASSERT_TRUE(compare_array(message_size, h_sent_messages[i].data(),
                                cpu_embedding.backward_sent_messages[i].data(), 1e-2));
    }
  }
};

/**************************** Test instantiations ****************************/

static const HybridEmbeddingConfig<uint32_t> config_uint32 = {4, 32, 10, 128, 1000, 128, 0.5f};
static const HybridEmbeddingConfig<long long> config_int64 = {4, 32, 10, 128, 1000, 128, 0.5f};

// Edge cases: no frequent, all frequent
static const HybridEmbeddingConfig<uint32_t> config_no_freq = {4, 32, 10, 128, 1000, 0, 0.5f};
static const HybridEmbeddingConfig<uint32_t> config_all_freq = {4, 32, 10, 128, 1000, 1000, 0.5f};

/* hybrid_embedding_forward_sent_message_test */

TEST(hybrid_embedding_forward_sent_message_test, uint32_half_64) {
  ForwardSentMessageTest<uint32_t, __half>(config_uint32, 64).run();
}

TEST(hybrid_embedding_forward_sent_message_test, int64_half_64) {
  ForwardSentMessageTest<long long, __half>(config_int64, 64).run();
}

TEST(hybrid_embedding_forward_sent_message_test, uint32_half_2048) {
  ForwardSentMessageTest<uint32_t, __half>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_forward_sent_message_test, int64_half_2048) {
  ForwardSentMessageTest<long long, __half>(config_int64, 2048).run();
}

TEST(hybrid_embedding_forward_sent_message_test, uint32_float_64) {
  ForwardSentMessageTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_forward_sent_message_test, int64_float_64) {
  ForwardSentMessageTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_forward_sent_message_test, uint32_float_2048) {
  ForwardSentMessageTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_forward_sent_message_test, int64_float_2048) {
  ForwardSentMessageTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_forward_sent_message_test, uint32_float_128_no_freq) {
  ForwardSentMessageTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_forward_sent_message_test, uint32_float_128_all_freq) {
  ForwardSentMessageTest<uint32_t, float>(config_all_freq, 128).run();
}

/* hybrid_embedding_backward_sent_message_test */

TEST(hybrid_embedding_backward_sent_message_test, uint32_half_64) {
  BackwardSentMessageTest<uint32_t, __half>(config_uint32, 64).run();
}

TEST(hybrid_embedding_backward_sent_message_test, int64_half_64) {
  BackwardSentMessageTest<long long, __half>(config_int64, 64).run();
}

TEST(hybrid_embedding_backward_sent_message_test, uint32_half_2048) {
  BackwardSentMessageTest<uint32_t, __half>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_backward_sent_message_test, int64_half_2048) {
  BackwardSentMessageTest<long long, __half>(config_int64, 2048).run();
}

TEST(hybrid_embedding_backward_sent_message_test, uint32_float_64) {
  BackwardSentMessageTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_backward_sent_message_test, int64_float_64) {
  BackwardSentMessageTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_backward_sent_message_test, uint32_float_2048) {
  BackwardSentMessageTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_backward_sent_message_test, int64_float_2048) {
  BackwardSentMessageTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_backward_sent_message_test, uint32_float_128_no_freq) {
  BackwardSentMessageTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_backward_sent_message_test, uint32_float_128_all_freq) {
  BackwardSentMessageTest<uint32_t, float>(config_all_freq, 128).run();
}
