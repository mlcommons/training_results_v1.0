/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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
#include <torch/extension.h>
#ifndef _syncfree_h_
#define _syncfree_h_ 

namespace at { namespace native { namespace syncfree {
    std::vector<torch::Tensor> pyt_nonzero_repeat(
            torch::Tensor input,
            torch::Tensor shuffle
            );
    std::vector<torch::Tensor> pyt_balanced_pos_neg_sampler_repeat(
	    torch::Tensor positive,
	    torch::Tensor positive_shuffle,
	    torch::Tensor negative,
	    torch::Tensor negative_shuffle,
	    long num_pos,
	    long batch_size,
            bool cat_outputs
	    );
    void pyt_index_fill(
	    torch::Tensor data,
	    long num_items,
	    torch::Tensor indices,
	    torch::Tensor counts,
	    long counts_idx,
	    bool cat_outputs,
	    torch::Tensor fill_value,
	    long fill_value_idx
	    );
    void pyt_index_fill_from(
	    torch::Tensor data,
	    long num_items,
	    torch::Tensor indices,
	    torch::Tensor counts,
	    long counts_idx,
	    bool cat_outputs,
	    torch::Tensor fill_values
	    );
} } }
#endif
