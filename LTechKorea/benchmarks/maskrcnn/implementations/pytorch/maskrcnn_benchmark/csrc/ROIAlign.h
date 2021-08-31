// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio,
                            const bool is_nhwc) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, is_nhwc);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio,
                             const bool is_nhwc) {
  if (grad.is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, is_nhwc);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor FourLevelsBatched_ROIAlign_forward(
		const at::Tensor& input_0,
		const at::Tensor& input_1,
		const at::Tensor& input_2,
		const at::Tensor& input_3,
		const at::Tensor& rois,
		const at::Tensor& rois_counts,
		const at::Tensor& level,
		const float spatial_scale_0,
		const float spatial_scale_1,
		const float spatial_scale_2,
		const float spatial_scale_3,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		const bool is_nhwc) {
    if (input_0.is_cuda()) {
#ifdef WITH_CUDA
      return FourLevelsBatched_ROIAlign_forward_cuda(
          input_0, input_1, input_2, input_3, rois, rois_counts, level, spatial_scale_0, spatial_scale_1, spatial_scale_2, spatial_scale_3,
          pooled_height, pooled_width, sampling_ratio, is_nhwc);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not supported on CPU");
}

std::vector<at::Tensor> FourLevelsBatched_ROIAlign_backward(
        const at::Tensor& grad,
        const at::Tensor& grad_counts,
        const at::Tensor& rois,
        const at::Tensor& level,
        const float spatial_scale_0,
        const float spatial_scale_1,
        const float spatial_scale_2,
        const float spatial_scale_3,
        const int pooled_height,
        const int pooled_width,
        const int batch_size,
        const int channels,
        const int height_0,
        const int height_1,
        const int height_2,
        const int height_3,
        const int width_0,
        const int width_1,
        const int width_2,
        const int width_3,
        const int sampling_ratio,
        const bool is_nhwc) {
    if (grad.is_cuda()) {
#ifdef WITH_CUDA
      return FourLevelsBatched_ROIAlign_backward_cuda(
          grad, grad_counts, rois, level, spatial_scale_0, spatial_scale_1, spatial_scale_2, spatial_scale_3, pooled_height, pooled_width, batch_size,
          channels, height_0, height_1, height_2, height_3, width_0, width_1, width_2, width_3, sampling_ratio, is_nhwc);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not supported on CPU");
}

