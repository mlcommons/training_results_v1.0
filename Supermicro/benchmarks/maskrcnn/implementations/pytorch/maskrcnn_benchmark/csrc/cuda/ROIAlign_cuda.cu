// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename U, typename T>
__device__ T bilinear_interpolate(const U* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename U, typename T>
__device__ void SingleSampleRoIAlignForward(
    const U* bottom_data, const T spatial_scale, const int height, const int width,  // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data,
    size_t index // per loop iteration
    )
{
  // (n, c, ph, pw) is an element in the pooled output
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;

  const T* offset_bottom_rois = bottom_rois + n * 5;
  int roi_batch_ind = offset_bottom_rois[0];

  // Do not using rounding; this implementation detail is critical
  T roi_start_w = offset_bottom_rois[1] * spatial_scale;
  T roi_start_h = offset_bottom_rois[2] * spatial_scale;
  T roi_end_w = offset_bottom_rois[3] * spatial_scale;
  T roi_end_h = offset_bottom_rois[4] * spatial_scale;
  // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
  // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
  // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
  // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

  // Force malformed ROIs to be 1x1
  T roi_width = max(roi_end_w - roi_start_w, (T)1.);
  T roi_height = max(roi_end_h - roi_start_h, (T)1.);
  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  const U* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

  T output_val = 0.;
  for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
  {
    const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix ++)
    {
      const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

      T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
      output_val += val;
    }
  }
  output_val /= count;

  top_data[index] = output_val;
}

// rois in math type (float). This is because ROIs come in as float. 
// TODO: Change other blocks producing ROI to support half type as well
template <typename U, typename T>
__global__ void RoIAlignForward(const int nthreads,
    const U* bottom_data, const T spatial_scale, const int height, const int width,  // per-level arguments
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    SingleSampleRoIAlignForward(
        bottom_data, spatial_scale, height, width,
        channels, pooled_height, pooled_width, sampling_ratio,
        bottom_rois, top_data,
        index);
  }
}

template <typename U, typename T, typename index_t>
__global__ void FourLevelsBatchedRoIAlignForward(const int nthreads, const int64_t* counts,
    const U* bottom_data_0, const T spatial_scale_0, const int height_0, const int width_0,  // per-level arguments
    const U* bottom_data_1, const T spatial_scale_1, const int height_1, const int width_1,
    const U* bottom_data_2, const T spatial_scale_2, const int height_2, const int width_2,
    const U* bottom_data_3, const T spatial_scale_3, const int height_3, const int width_3,
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data,
    const index_t* level
    )
{
  int counts_threads = (counts != NULL) ? counts[0]*pooled_width*pooled_height*channels : nthreads;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (index < counts_threads) {
      int n = index / pooled_width / pooled_height / channels;
      switch (level[n]) {
	case 0:
	  SingleSampleRoIAlignForward(
	      bottom_data_0, spatial_scale_0, height_0, width_0,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	case 1:
	  SingleSampleRoIAlignForward(
	      bottom_data_1, spatial_scale_1, height_1, width_1,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	case 2:
	  SingleSampleRoIAlignForward(
	      bottom_data_2, spatial_scale_2, height_2, width_2,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	case 3:
	  SingleSampleRoIAlignForward(
	      bottom_data_3, spatial_scale_3, height_3, width_3,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	default:
	  break;
      }
    } else {
      top_data[index] = 0;
    }
  }
}

template <typename U, typename T>
__device__ T bilinear_interpolate_nhwc(const U* bottom_data,

    const int height, const int width, const int channels,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[channels * (y_low * width + x_low)];
  T v2 = bottom_data[channels * (y_low * width + x_high)];
  T v3 = bottom_data[channels * (y_high * width + x_low)];
  T v4 = bottom_data[channels * (y_high * width + x_high)];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename U, typename T>
__device__ void SingleSampleRoIAlignForwardNHWC(
    const U* bottom_data, const T spatial_scale, const int height, const int width,  // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data,
    size_t index  // per loop iteration
    ) 
{
    // (n, c, ph, pw) is an element in the pooled output
    int c = index % channels;
    int pw = (index / channels) % pooled_width;
    int ph = (index / channels / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const U* offset_bottom_data = bottom_data + (roi_batch_ind * channels * height * width + c);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate_nhwc(offset_bottom_data, height, width, channels, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
}

// rois in math type (float). This is because ROIs come in as float. 
// TODO: Change other blocks producing ROI to support half type as well
template <typename U, typename T>
__global__ void RoIAlignForwardNHWC(const int nthreads,
    const U* bottom_data, const T spatial_scale, const int height, const int width, // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    SingleSampleRoIAlignForwardNHWC(
        bottom_data, spatial_scale, height, width,
        channels, pooled_height, pooled_width, sampling_ratio,
        bottom_rois, top_data,
        index);
  }
}

template <typename U, typename T, typename index_t>
__global__ void FourLevelsBatchedRoIAlignForwardNHWC(const int nthreads, const int64_t* counts,
    const U* bottom_data_0, const T spatial_scale_0, const int height_0, const int width_0,
    const U* bottom_data_1, const T spatial_scale_1, const int height_1, const int width_1,
    const U* bottom_data_2, const T spatial_scale_2, const int height_2, const int width_2,
    const U* bottom_data_3, const T spatial_scale_3, const int height_3, const int width_3,
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois, U* top_data,
    const index_t* level) 
{
  int counts_threads = (counts != NULL) ? counts[0]*pooled_width*pooled_height*channels : nthreads;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (index < counts_threads) {
      int n = index / pooled_width / pooled_height / channels;
      switch (level[n]) {
	case 0:
	  SingleSampleRoIAlignForwardNHWC(
	      bottom_data_0, spatial_scale_0, height_0, width_0,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	  break;
	case 1:
	  SingleSampleRoIAlignForwardNHWC(
	      bottom_data_1, spatial_scale_1, height_1, width_1,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	  break;
	case 2:
	  SingleSampleRoIAlignForwardNHWC(
	      bottom_data_2, spatial_scale_2, height_2, width_2,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	  break;
	case 3:
	  SingleSampleRoIAlignForwardNHWC(
	      bottom_data_3, spatial_scale_3, height_3, width_3,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois, top_data,
	      index);
	  break;
	default:
	  break;
      }
    } else {
      top_data[index] = 0;
    }
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename U, typename T>
__device__ void SingleSampleRoIAlignBackwardFeature(
    const U* top_diff, 
    const T spatial_scale, const int height, const int width, U* bottom_diff,   // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois,
    size_t index  // per loop iteration
    )
{
  // (n, c, ph, pw) is an element in the pooled output
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;

  const T* offset_bottom_rois = bottom_rois + n * 5;
  int roi_batch_ind = offset_bottom_rois[0];

  // Do not using rounding; this implementation detail is critical
  T roi_start_w = offset_bottom_rois[1] * spatial_scale;
  T roi_start_h = offset_bottom_rois[2] * spatial_scale;
  T roi_end_w = offset_bottom_rois[3] * spatial_scale;
  T roi_end_h = offset_bottom_rois[4] * spatial_scale;
  // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
  // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
  // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
  // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

  // Force malformed ROIs to be 1x1
  T roi_width = max(roi_end_w - roi_start_w, (T)1.);
  T roi_height = max(roi_end_h - roi_start_h, (T)1.);
  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  U* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

  int top_offset    = (n * channels + c) * pooled_height * pooled_width;
  const U* offset_top_diff = top_diff + top_offset;
  const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

  for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
  {
    const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix ++)
    {
      const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

      T w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;

      bilinear_interpolate_gradient(height, width, y, x,
	  w1, w2, w3, w4,
	  x_low, x_high, y_low, y_high,
	  index);

      T g1 = top_diff_this_bin * w1 / count;
      T g2 = top_diff_this_bin * w2 / count;
      T g3 = top_diff_this_bin * w3 / count;
      T g4 = top_diff_this_bin * w4 / count;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
      {
	atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
	atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
	atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
	atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
      } // if
    } // ix
  } // iy
}

template <typename U, typename T>
__global__ void RoIAlignBackwardFeature(const int nthreads, const U* top_diff, 
    const T spatial_scale, const int height, const int width, U* bottom_diff,   // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois
    )
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    SingleSampleRoIAlignBackwardFeature(top_diff,
        spatial_scale, height, width, bottom_diff,
        channels, pooled_height, pooled_width, sampling_ratio,
        bottom_rois,
        index);
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

template <typename U, typename T, typename index_t>
__global__ void FourLevelsBatchedRoIAlignBackwardFeature(const int nthreads, const int64_t* counts, 
    const U* top_diff,
    const T spatial_scale_0, const int height_0, const int width_0, U* bottom_diff_0,   // per level
    const T spatial_scale_1, const int height_1, const int width_1, U* bottom_diff_1,
    const T spatial_scale_2, const int height_2, const int width_2, U* bottom_diff_2,
    const T spatial_scale_3, const int height_3, const int width_3, U* bottom_diff_3,
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois,
    const index_t* level
    )
{
  int counts_threads = (counts != NULL) ? counts[0]*pooled_width*pooled_height*channels : nthreads;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (index < counts_threads) {
      int n = index / pooled_width / pooled_height / channels;
      switch (level[n]) {
	case 0:
	  SingleSampleRoIAlignBackwardFeature(top_diff,
	      spatial_scale_0, height_0, width_0, bottom_diff_0,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 1:
	  SingleSampleRoIAlignBackwardFeature(top_diff,
	      spatial_scale_1, height_1, width_1, bottom_diff_1,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 2:
	  SingleSampleRoIAlignBackwardFeature(top_diff,
	      spatial_scale_2, height_2, width_2, bottom_diff_2,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 3:
	  SingleSampleRoIAlignBackwardFeature(top_diff,
	      spatial_scale_3, height_3, width_3, bottom_diff_3,
	      channels, pooled_height, pooled_width, sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	default:
	  break;
      }
    }
  } // CUDA_1D_KERNEL_LOOP
}

template <typename U, typename T>
__device__ void SingleSampleRoIAlignBackwardFeatureNHWC(const U* top_diff,
    const T spatial_scale, const int height, const int width, U* bottom_diff,   // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois,
    size_t index  // per loop iteration
    )
{
  // (n, c, ph, pw) is an element in the pooled output
  int c = index % channels;
  int pw = (index / channels) % pooled_width;
  int ph = (index / channels / pooled_width) % pooled_height;
  int n = index / pooled_width / pooled_height / channels;

  const T* offset_bottom_rois = bottom_rois + n * 5;
  int roi_batch_ind = offset_bottom_rois[0];

  // Do not using rounding; this implementation detail is critical
  T roi_start_w = offset_bottom_rois[1] * spatial_scale;
  T roi_start_h = offset_bottom_rois[2] * spatial_scale;
  T roi_end_w = offset_bottom_rois[3] * spatial_scale;
  T roi_end_h = offset_bottom_rois[4] * spatial_scale;
  // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
  // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
  // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
  // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

  // Force malformed ROIs to be 1x1
  T roi_width = max(roi_end_w - roi_start_w, (T)1.);
  T roi_height = max(roi_end_h - roi_start_h, (T)1.);
  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  U* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels * height * width + c);

  int top_offset    = n * channels * pooled_height * pooled_width + c;
  const U* offset_top_diff = top_diff + top_offset;
  const T top_diff_this_bin = offset_top_diff[channels * (ph * pooled_width + pw)];

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

  for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
  {
    const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix ++)
    {
      const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

      T w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;

      bilinear_interpolate_gradient(height, width, y, x,
	  w1, w2, w3, w4,
	  x_low, x_high, y_low, y_high,
	  index);

      T g1 = top_diff_this_bin * w1 / count;
      T g2 = top_diff_this_bin * w2 / count;
      T g3 = top_diff_this_bin * w3 / count;
      T g4 = top_diff_this_bin * w4 / count;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
      {
	atomicAdd(offset_bottom_diff + channels * (y_low * width + x_low), static_cast<T>(g1));
	atomicAdd(offset_bottom_diff + channels * (y_low * width + x_high), static_cast<T>(g2));
	atomicAdd(offset_bottom_diff + channels * (y_high * width + x_low), static_cast<T>(g3));
	atomicAdd(offset_bottom_diff + channels * (y_high * width + x_high), static_cast<T>(g4));
      } // if
    } // ix
  } // iy
}

template <typename U, typename T>
__global__ void RoIAlignBackwardFeatureNHWC(const int nthreads, const U* top_diff,
    const T spatial_scale, const int height, const int width, U* bottom_diff,   // per level
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois
    )
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    SingleSampleRoIAlignBackwardFeatureNHWC(top_diff,
	spatial_scale,height,width,bottom_diff,
	channels,pooled_height,pooled_width,sampling_ratio,
	bottom_rois,
	index);
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

template <typename U, typename T, typename index_t>
__global__ void FourLevelsBatchedRoIAlignBackwardFeatureNHWC(const int nthreads, const int64_t* counts,
    const U* top_diff,
    const T spatial_scale_0, const int height_0, const int width_0, U* bottom_diff_0,   // per level
    const T spatial_scale_1, const int height_1, const int width_1, U* bottom_diff_1,
    const T spatial_scale_2, const int height_2, const int width_2, U* bottom_diff_2,
    const T spatial_scale_3, const int height_3, const int width_3, U* bottom_diff_3,
    const int channels, const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* bottom_rois,
    index_t* level
    )
{
  int counts_threads = (counts != NULL) ? counts[0]*pooled_width*pooled_height*channels : nthreads;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (index < counts_threads) {
      int n = index / pooled_width / pooled_height / channels;
      switch (level[n]) {
	case 0:
	  SingleSampleRoIAlignBackwardFeatureNHWC(top_diff,
	      spatial_scale_0,height_0,width_0,bottom_diff_0,
	      channels,pooled_height,pooled_width,sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 1:
	  SingleSampleRoIAlignBackwardFeatureNHWC(top_diff,
	      spatial_scale_1,height_1,width_1,bottom_diff_1,
	      channels,pooled_height,pooled_width,sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 2:
	  SingleSampleRoIAlignBackwardFeatureNHWC(top_diff,
	      spatial_scale_2,height_2,width_2,bottom_diff_2,
	      channels,pooled_height,pooled_width,sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	case 3:
	  SingleSampleRoIAlignBackwardFeatureNHWC(top_diff,
	      spatial_scale_3,height_3,width_3,bottom_diff_3,
	      channels,pooled_height,pooled_width,sampling_ratio,
	      bottom_rois,
	      index);
	  break;
	default:
	  break;
      }
    }
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio,
				 const bool is_nhwc) {
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = is_nhwc ? input.size(3) : input.size(1);
  auto height   = is_nhwc ? input.size(1) : input.size(2);
  auto width    = is_nhwc ? input.size(2) : input.size(3);

  auto output = is_nhwc ? at::empty({num_rois, pooled_height, pooled_width, channels}, input.options()) : 
	  at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  int gridSize;
  int blockSize;
  cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                     &blockSize,
                                     (void*) RoIAlignForward<float, float>,
                                     0,  // dynamic memory
                                     0); // maximum utilized threads   

  dim3 grid(gridSize);
  dim3 block(blockSize);
  
  //TODO: Math type is hard coded to float assuming double is not used, if needed, add a case for double as well. 
  //In case of double, it should be <double, double>, not <double, float>
  //TODO: ROIs come in as float, fix other blocks so they come in as same type as input. 
  if (!is_nhwc){
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ROIAlign_forward", [&] {
      RoIAlignForward<scalar_t, float><<<grid, block, 0, stream>>>(
           output_size,
           input.contiguous().data_ptr<scalar_t>(),
           spatial_scale,
           height,
           width,
           channels,
           pooled_height,
           pooled_width,
           sampling_ratio,
           rois.contiguous().data_ptr<float>(),
           output.data_ptr<scalar_t>());
    });
  }
  else{
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ROIAlign_forward", [&] {
      RoIAlignForwardNHWC<scalar_t, float><<<grid, block, 0, stream>>>(
           output_size,
           input.contiguous().data_ptr<scalar_t>(),
           spatial_scale,
           height,
           width,
           channels,
           pooled_height,
           pooled_width,
           sampling_ratio,
           rois.contiguous().data_ptr<float>(),
           output.data_ptr<scalar_t>());
    });
  }
  THCudaCheck(cudaGetLastError());
  return output;
}

at::Tensor FourLevelsBatched_ROIAlign_forward_cuda(
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
  AT_ASSERTM(input_0.is_cuda(), "input_0 must be a CUDA tensor");
  AT_ASSERTM(input_1.is_cuda(), "input_1 must be a CUDA tensor");
  AT_ASSERTM(input_2.is_cuda(), "input_2 must be a CUDA tensor");
  AT_ASSERTM(input_3.is_cuda(), "input_3 must be a CUDA tensor");
  AT_ASSERTM(input_0.device() == input_1.device() && input_0.device() == input_2.device() && input_0.device() == input_3.device(), "input_* must all be on same device");
  AT_ASSERTM(input_0.dtype() == input_1.dtype() && input_0.dtype() == input_2.dtype() && input_0.dtype() == input_3.dtype(), "input_* must all be same dtype");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");
  AT_ASSERTM(rois_counts.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels_0 = is_nhwc ? input_0.size(3) : input_0.size(1);
  auto height_0   = is_nhwc ? input_0.size(1) : input_0.size(2);
  auto width_0    = is_nhwc ? input_0.size(2) : input_0.size(3);
  auto channels_1 = is_nhwc ? input_1.size(3) : input_1.size(1);
  auto height_1   = is_nhwc ? input_1.size(1) : input_1.size(2);
  auto width_1    = is_nhwc ? input_1.size(2) : input_1.size(3);
  auto channels_2 = is_nhwc ? input_2.size(3) : input_2.size(1);
  auto height_2   = is_nhwc ? input_2.size(1) : input_2.size(2);
  auto width_2    = is_nhwc ? input_2.size(2) : input_2.size(3);
  auto channels_3 = is_nhwc ? input_3.size(3) : input_3.size(1);
  auto height_3   = is_nhwc ? input_3.size(1) : input_3.size(2);
  auto width_3    = is_nhwc ? input_3.size(2) : input_3.size(3);
  AT_ASSERT(channels_0 == channels_1 && channels_0 == channels_2 && channels_0 == channels_3, "Channel counts differ");
  auto channels = channels_0;

  auto output = is_nhwc ? at::empty({num_rois, pooled_height, pooled_width, channels}, input_0.options()) : 
	  at::empty({num_rois, channels, pooled_height, pooled_width}, input_0.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  //TODO: Math type is hard coded to float assuming double is not used, if needed, add a case for double as well. 
  //In case of double, it should be <double, double>, not <double, float>
  //TODO: ROIs come in as float, fix other blocks so they come in as same type as input. 
  if (!is_nhwc){
    AT_DISPATCH_INDEX_TYPES(level.scalar_type(), "ROIAlign_forward", [&] {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_0.scalar_type(), "ROIAlign_forward", [&] {
        int gridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                           &blockSize,
                                           (void*) FourLevelsBatchedRoIAlignForward<scalar_t, float, index_t>,
                                           0,  // dynamic memory
                                           0); // maximum utilized threads   
        dim3 grid(gridSize);
        dim3 block(blockSize);
        FourLevelsBatchedRoIAlignForward<scalar_t, float, index_t><<<grid, block, 0, stream>>>(
             output_size,
	     rois_counts.numel() == 0 ? NULL : rois_counts.data_ptr<int64_t>(),
             input_0.contiguous().data_ptr<scalar_t>(), spatial_scale_0, height_0, width_0,
             input_1.contiguous().data_ptr<scalar_t>(), spatial_scale_1, height_1, width_1,
             input_2.contiguous().data_ptr<scalar_t>(), spatial_scale_2, height_2, width_2,
             input_3.contiguous().data_ptr<scalar_t>(), spatial_scale_3, height_3, width_3,
             channels,
             pooled_height,
             pooled_width,
             sampling_ratio,
             rois.contiguous().data_ptr<float>(),
             output.data_ptr<scalar_t>(),
  	     level.contiguous().data_ptr<index_t>());
      });
    });
  }
  else{
    AT_DISPATCH_INDEX_TYPES(level.scalar_type(), "ROIAlign_forward", [&] {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_0.scalar_type(), "ROIAlign_forward", [&] {
        int gridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                           &blockSize,
                                           (void*) FourLevelsBatchedRoIAlignForwardNHWC<scalar_t, float, index_t>,
                                           0,  // dynamic memory
                                           0); // maximum utilized threads   
        dim3 grid(gridSize);
        dim3 block(blockSize);
        FourLevelsBatchedRoIAlignForwardNHWC<scalar_t, float, index_t><<<grid, block, 0, stream>>>(
             output_size,
	     rois_counts.numel() == 0 ? NULL : rois_counts.data_ptr<int64_t>(),
             input_0.contiguous().data_ptr<scalar_t>(), spatial_scale_0, height_0, width_0,
             input_1.contiguous().data_ptr<scalar_t>(), spatial_scale_1, height_1, width_1,
             input_2.contiguous().data_ptr<scalar_t>(), spatial_scale_2, height_2, width_2,
             input_3.contiguous().data_ptr<scalar_t>(), spatial_scale_3, height_3, width_3,
             channels,
             pooled_height,
             pooled_width,
             sampling_ratio,
             rois.contiguous().data_ptr<float>(),
             output.data_ptr<scalar_t>(),
  	     level.contiguous().data_ptr<index_t>());
      });
    });
  }
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
// NHWC + layout transposes are faster than NCHW, so just keep the NHWC implementation for backward pass
at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
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
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = is_nhwc ? at::zeros({batch_size, height, width, channels}, grad.options()) : 
	  at::zeros({batch_size, channels, height, width}, grad.options());


  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  int gridSize;
  int blockSize;
  cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                     &blockSize,
                                     (void*) RoIAlignBackwardFeature<float, float>,
                                     0,  // dynamic memory
                                     0); // maximum utilized threads   

  dim3 grid(gridSize);
  dim3 block(blockSize);
  
  //TODO: Math type is hard coded to float assuming double is not used, if needed, add a case for double as well. 
  //In case of double, it should be <double, double>, not <double, float>
  //TODO: ROIs come in as float, fix other blocks so they come in as same type as input. 
  if (!is_nhwc){
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "ROIAlign_backward", [&] {
      RoIAlignBackwardFeature<scalar_t, float><<<grid, block, 0, stream>>>(
           grad.numel(),
           grad.contiguous().data_ptr<scalar_t>(),
           spatial_scale,
           height,
           width,
           grad_input.data_ptr<scalar_t>(),
           channels,
           pooled_height,
           pooled_width,
           sampling_ratio,
           rois.contiguous().data_ptr<float>());
    });
  }
  else{
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "ROIAlign_backward", [&] {
      RoIAlignBackwardFeatureNHWC<scalar_t, float><<<grid, block, 0, stream>>>(
           grad.numel(),
           grad.contiguous().data_ptr<scalar_t>(),
           spatial_scale,
           height,
           width,
           grad_input.data_ptr<scalar_t>(),
           channels,
           pooled_height,
           pooled_width,
           sampling_ratio,
           rois.contiguous().data_ptr<float>());
    });
  }
  THCudaCheck(cudaGetLastError());
  return grad_input;
}

std::vector<at::Tensor> FourLevelsBatched_ROIAlign_backward_cuda(const at::Tensor& grad,
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
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(grad_counts.is_cuda(), "grad_counts must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input_0 = is_nhwc ? at::zeros({batch_size, height_0, width_0, channels}, grad.options()) : at::zeros({batch_size, channels, height_0, width_0}, grad.options());
  auto grad_input_1 = is_nhwc ? at::zeros({batch_size, height_1, width_1, channels}, grad.options()) : at::zeros({batch_size, channels, height_1, width_1}, grad.options());
  auto grad_input_2 = is_nhwc ? at::zeros({batch_size, height_2, width_2, channels}, grad.options()) : at::zeros({batch_size, channels, height_2, width_2}, grad.options());
  auto grad_input_3 = is_nhwc ? at::zeros({batch_size, height_3, width_3, channels}, grad.options()) : at::zeros({batch_size, channels, height_3, width_3}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return {grad_input_0, grad_input_1, grad_input_2, grad_input_3};
  }

  //TODO: Math type is hard coded to float assuming double is not used, if needed, add a case for double as well. 
  //In case of double, it should be <double, double>, not <double, float>
  //TODO: ROIs come in as float, fix other blocks so they come in as same type as input. 
  if (!is_nhwc){
    AT_DISPATCH_INDEX_TYPES(level.scalar_type(), "ROIAlign_backward", [&] {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "ROIAlign_backward", [&] {
        int gridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                           &blockSize,
                                           (void*) FourLevelsBatchedRoIAlignBackwardFeature<scalar_t, float, index_t>,
                                           0,  // dynamic memory
                                           0); // maximum utilized threads   
        dim3 grid(gridSize);
        dim3 block(blockSize);
        FourLevelsBatchedRoIAlignBackwardFeature<scalar_t, float, index_t><<<grid, block, 0, stream>>>(
             grad.numel(),
	     grad_counts.numel() == 0 ? NULL : grad_counts.data_ptr<int64_t>(),
             grad.contiguous().data_ptr<scalar_t>(),
             spatial_scale_0, height_0, width_0, grad_input_0.data_ptr<scalar_t>(),
             spatial_scale_1, height_1, width_1, grad_input_1.data_ptr<scalar_t>(),
             spatial_scale_2, height_2, width_2, grad_input_2.data_ptr<scalar_t>(),
             spatial_scale_3, height_3, width_3, grad_input_3.data_ptr<scalar_t>(),
             channels, pooled_height, pooled_width, sampling_ratio,
             rois.contiguous().data_ptr<float>(),
             level.contiguous().data_ptr<index_t>());
      });
    });
  }
  else{
    AT_DISPATCH_INDEX_TYPES(level.scalar_type(), "ROIAlign_backward", [&] {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "ROIAlign_backward", [&] {
        int gridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&gridSize,
                                           &blockSize,
                                           (void*) FourLevelsBatchedRoIAlignBackwardFeatureNHWC<scalar_t, float, index_t>,
                                           0,  // dynamic memory
                                           0); // maximum utilized threads   
        dim3 grid(gridSize);
        dim3 block(blockSize);
        FourLevelsBatchedRoIAlignBackwardFeatureNHWC<scalar_t, float, index_t><<<grid, block, 0, stream>>>(
             grad.numel(),
	     grad_counts.numel() == 0 ? NULL : grad_counts.data_ptr<int64_t>(),
             grad.contiguous().data_ptr<scalar_t>(),
             spatial_scale_0, height_0, width_0, grad_input_0.data_ptr<scalar_t>(),
             spatial_scale_1, height_1, width_1, grad_input_1.data_ptr<scalar_t>(),
             spatial_scale_2, height_2, width_2, grad_input_2.data_ptr<scalar_t>(),
             spatial_scale_3, height_3, width_3, grad_input_3.data_ptr<scalar_t>(),
             channels, pooled_height, pooled_width, sampling_ratio,
             rois.contiguous().data_ptr<float>(),
             level.contiguous().data_ptr<index_t>());
      });
    });
  }
  THCudaCheck(cudaGetLastError());
  return {grad_input_0, grad_input_1, grad_input_2, grad_input_3};
}
