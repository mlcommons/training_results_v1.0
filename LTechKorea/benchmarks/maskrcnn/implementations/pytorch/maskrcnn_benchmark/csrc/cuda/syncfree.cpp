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

#include "syncfree.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nonzero_repeat", &at::native::syncfree::pyt_nonzero_repeat, "nonzero_repeat");
  m.def("balanced_pos_neg_sampler_repeat", &at::native::syncfree::pyt_balanced_pos_neg_sampler_repeat, "balanced_pos_neg_sampler_repeat");
  m.def("index_fill", &at::native::syncfree::pyt_index_fill, "index_fill");
  m.def("index_fill_from", &at::native::syncfree::pyt_index_fill_from, "index_fill_from");
}
