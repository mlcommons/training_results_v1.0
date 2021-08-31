#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cassert>

/* 
 * NB! This namespace prefix is necessary in order to prevent CUB used in MRCNN from
 *     messing with CUB used by Thrust in Pytorch upstream code.
 *     CUB relies on some global variables.
 */
#define CUB_NS_PREFIX namespace mrcnn {
#define CUB_NS_POSTFIX }
#include <cub/cub.cuh>

namespace at { namespace native { namespace syncfree {

void cuPrint(const char* msg, int* items, int num_items)
{
    int* h_items = new int[num_items];
    cudaMemcpy(h_items, items, num_items*sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s :: %d", msg, h_items[0]);
    for (int i = 1;  i < num_items;  ++i) printf(", %d", h_items[i]);
    printf("\n");
    delete [] h_items;
}

void cuPrint(const char* msg, float* items, int num_items)
{
    float* h_items = new float[num_items];
    cudaMemcpy(h_items, items, num_items*sizeof(float), cudaMemcpyDeviceToHost);
    printf("%s :: %f", msg, h_items[0]);
    for (int i = 1;  i < num_items;  ++i) printf(", %f", h_items[i]);
    printf("\n");
    delete [] h_items;
}

template<typename T_f, typename T_i>
__global__ void find_nonzero(T_f* d_values, T_i* d_shuffle, T_i num_items, T_i* d_out)
{
    T_i thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < num_items) {
        T_i shuf_idx = (d_shuffle != NULL) ? d_shuffle[thid] : thid;
        d_out[thid] = (d_values[shuf_idx] != 0) ? 1 : 0;
    }
}

template<typename T_f, typename T_i>
__global__ void write_nonzero(T_f* d_values, T_i* d_shuffle, T_i num_items, T_i num_outputs, bool cat_outputs, T_i* d_offsets, T_i* d_indices, T_i* d_counts, T_i counts_idx)
{
    T_i thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < num_items) {
        T_i offset = d_offsets[thid];
        T_i glob_offset = (counts_idx > 0) ? offset + d_counts[counts_idx-1] : offset;
        T_i shuf_idx = (d_shuffle != NULL) ? d_shuffle[thid] : thid;
        T_i is_nonzero = (d_values[shuf_idx] != 0) ? 1 : 0;
        if (glob_offset >= num_outputs) {
            glob_offset = num_outputs;
	    is_nonzero = 0;
        } else if (is_nonzero) {
            d_indices[cat_outputs?glob_offset:offset] = shuf_idx;
        }
        if (thid == num_items-1) {
            d_counts[counts_idx] = glob_offset + is_nonzero;
        }
    }
}

template<typename T_i>
__global__ void repeat(T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs)
{
    T_i n = (cat_outputs || counts_idx == 0) ? d_counts[counts_idx] : d_counts[counts_idx] - d_counts[counts_idx-1];
    if (n > 0) {
	T_i thid = blockIdx.x * blockDim.x + threadIdx.x + n;
	if (thid < num_items) {
	    d_indices[thid] = d_indices[thid%n];
	}
    }
}

void check_cuda_error(int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
	printf("CUDA ERROR on line %d: %s\n", line, cudaGetErrorString(err));
    }
}

template<typename T_f, typename T_i>
void nonzero(cudaStream_t stream, T_f* d_values, T_i* d_shuffle, T_i num_items, T_i num_outputs, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs, T_i* d_offsets, void* d_temp_storage, size_t temp_storage_bytes)
{
    int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid((num_items+numThreads-1)/numThreads, 1, 1);

    // compute offsets for nonzero outputs
    find_nonzero<<<dimGrid, dimBlock, 0, stream>>>(d_values, d_shuffle, num_items, d_offsets);
    check_cuda_error(__LINE__);

    mrcnn::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_offsets, d_offsets, num_items, stream);
    check_cuda_error(__LINE__);
    
    // write out nonzero items
    write_nonzero<<<dimGrid, dimBlock, 0, stream>>>(d_values, d_shuffle, num_items, num_outputs, cat_outputs, d_offsets, d_indices, d_counts, counts_idx);
    check_cuda_error(__LINE__);
}

template<typename T_i>
void repeat(cudaStream_t stream, T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs)
{
    int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid((num_items+numThreads-1)/numThreads, 1, 1);

    // repeat nonzero indexes
    repeat<<<dimGrid, dimBlock, 0, stream>>>(num_items, d_indices, d_counts, counts_idx, cat_outputs);
    check_cuda_error(__LINE__);
}

template<typename T_f, typename T_i>
__global__ void index_fill(T_f* d_values, T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs, T_f* fill_value, T_i fill_value_idx)
{
    T_i n0 = (counts_idx > 0) ? d_counts[counts_idx-1] : 0;
    T_i nn = d_counts[counts_idx] - n0;
    T_i thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < nn) {
	T_i index = cat_outputs ? d_indices[n0+thid] : d_indices[thid];
	d_values[index] = fill_value[fill_value_idx];
    }
}

template<typename T_f, typename T_i>
__global__ void index_fill_from(T_f* d_values, T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs, T_f* fill_values)
{
    T_i n0 = (counts_idx > 0) ? d_counts[counts_idx-1] : 0;
    T_i nn = d_counts[counts_idx] - n0;
    T_i thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < nn) {
	T_i index = cat_outputs ? d_indices[n0+thid] : d_indices[thid];
	d_values[index] = fill_values[index];
    }
}

template<typename T_f, typename T_i>
void index_fill(cudaStream_t stream, T_f* d_values, T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs, T_f* fill_value, T_i fill_value_idx)
{
    int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid((num_items+numThreads-1)/numThreads, 1, 1);

    index_fill<<<dimGrid, dimBlock, 0, stream>>>(d_values, num_items, d_indices, d_counts, counts_idx, cat_outputs, fill_value, fill_value_idx);
    check_cuda_error(__LINE__);
}

template<typename T_f, typename T_i>
void index_fill_from(cudaStream_t stream, T_f* d_values, T_i num_items, T_i* d_indices, T_i* d_counts, T_i counts_idx, bool cat_outputs, T_f* fill_values)
{
    int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid((num_items+numThreads-1)/numThreads, 1, 1);

    index_fill_from<<<dimGrid, dimBlock, 0, stream>>>(d_values, num_items, d_indices, d_counts, counts_idx, cat_outputs, fill_values);
    check_cuda_error(__LINE__);
}

std::vector<torch::Tensor> pyt_nonzero_repeat(
        torch::Tensor input,
        torch::Tensor shuffle
	)
{
    auto indices = torch::empty_like(input, shuffle.options());
    auto counts = torch::zeros({1}, indices.options());
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "pyt_nonzero_repeat", [&]() {
	    index_t n = static_cast<index_t>(input.numel());
            auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
	    auto offsets = allocator.allocate(n*sizeof(index_t));
	    index_t* offsets_ptr = static_cast<index_t*>(offsets.get());

            size_t temp_storage_bytes = 0;
            cudaError_t err = mrcnn::cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, indices.data_ptr<index_t>(), offsets_ptr, n);
            if (err != cudaSuccess) {
                printf("mrcnn::cub::DeviceScan::ExclusiveSum returned %s\n", cudaGetErrorName(err));
            }
            auto temp_storage = allocator.allocate(temp_storage_bytes);

            AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "pyt_nonzero_repeat", [&]() {
                nonzero<scalar_t, index_t>(
                        at::cuda::getCurrentCUDAStream(),
                        input.contiguous().data_ptr<scalar_t>(),
                        shuffle.numel() > 0 ? shuffle.contiguous().data_ptr<index_t>() : NULL,
			n, n,
                        indices.data_ptr<index_t>(),
                        counts.data_ptr<index_t>(),
                        0, false,
			offsets_ptr,
                        temp_storage.get(),
                        temp_storage_bytes
			); 
                repeat<index_t>(
                        at::cuda::getCurrentCUDAStream(),
                        n,
                        indices.data_ptr<index_t>(),
                        counts.data_ptr<index_t>(),
                        0, false);
		});
    });

    return {indices, counts};
}

/*
 * Collect at most num_pos nonzero samples from positive.
 * Collect at most batch_size - min(num_pos, nonzero(positive).numel()) samples from negative.
 */
std::vector<torch::Tensor> pyt_balanced_pos_neg_sampler_repeat(
        torch::Tensor positive,
        torch::Tensor positive_shuffle,
        torch::Tensor negative,
        torch::Tensor negative_shuffle,
        long num_pos,
        long batch_size,
        bool cat_outputs
        )
{
    torch::Tensor indices, indices2;
    if (cat_outputs) {
        indices = torch::empty({batch_size}, positive_shuffle.options());
    } else {
        indices = torch::empty({num_pos}, positive_shuffle.options());
        indices2 = torch::empty({batch_size}, positive_shuffle.options());
    }
    auto counts = torch::zeros({2}, indices.options());
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "pyt_balanced_pos_neg_sampler", [&]() {
        index_t np = static_cast<index_t>(positive.numel());
        index_t nn = static_cast<index_t>(negative.numel());
        index_t nmax = np > nn ? np : nn;
        auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
        auto offsets = allocator.allocate(nmax*sizeof(index_t));
        index_t* offsets_ptr = static_cast<index_t*>(offsets.get());

        size_t temp_storage_bytes_np = 0, temp_storage_bytes_nn = 0;
        cudaError_t err = mrcnn::cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes_np, offsets_ptr, offsets_ptr, np);
        if (err != cudaSuccess) printf("mrcnn::cub::DeviceScan::ExclusiveSum returned %s\n", cudaGetErrorName(err));
        err = mrcnn::cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes_nn, offsets_ptr, offsets_ptr, nn);
        if (err != cudaSuccess) printf("mrcnn::cub::DeviceScan::ExclusiveSum returned %s\n", cudaGetErrorName(err));
        size_t temp_storage_bytes = temp_storage_bytes_np > temp_storage_bytes_nn ? temp_storage_bytes_np : temp_storage_bytes_nn;
        auto temp_storage = allocator.allocate(temp_storage_bytes);

        //printf("temp_storage_bytes = %ld, num_pos = %ld, batch_size = %ld, np = %d, nn = %d\n", temp_storage_bytes, num_pos, batch_size, np, nn);

        AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, positive.scalar_type(), "pyt_balanced_pos_neg_sampler", [&]() {
            nonzero<scalar_t, index_t>(
                at::cuda::getCurrentCUDAStream(),
                positive.contiguous().data_ptr<scalar_t>(),
                positive_shuffle.numel() > 0 ? positive_shuffle.contiguous().data_ptr<index_t>() : nullptr,
                np, num_pos,
                indices.data_ptr<index_t>(),
                counts.data_ptr<index_t>(),
                0, cat_outputs,
                offsets_ptr,
                temp_storage.get(),
                temp_storage_bytes_np
                ); 
            if (!cat_outputs) {
                repeat<index_t>(
                    at::cuda::getCurrentCUDAStream(),
                    num_pos,
                    indices.data_ptr<index_t>(),
                    counts.data_ptr<index_t>(),
                    0, cat_outputs);
            }
            nonzero<scalar_t, index_t>(
                at::cuda::getCurrentCUDAStream(),
                negative.contiguous().data_ptr<scalar_t>(),
                negative_shuffle.numel() > 0 ? negative_shuffle.contiguous().data_ptr<index_t>() : nullptr,
                nn, batch_size,
                cat_outputs ? indices.data_ptr<index_t>() : indices2.data_ptr<index_t>(),
                counts.data_ptr<index_t>(),
                1, cat_outputs,
                offsets_ptr,
                temp_storage.get(),
                temp_storage_bytes_nn
                ); 
            repeat<index_t>(
                at::cuda::getCurrentCUDAStream(),
                batch_size,
                cat_outputs ? indices.data_ptr<index_t>() : indices2.data_ptr<index_t>(),
                counts.data_ptr<index_t>(),
                1, cat_outputs);
        });
    });

    if (cat_outputs) {
        return {indices, counts};
    } else {
        return {indices, indices2, counts};
    }
}

void pyt_index_fill(
	torch::Tensor data,
	long num_items,
	torch::Tensor indices,
	torch::Tensor counts,
	long counts_idx,
	bool cat_outputs,
	torch::Tensor fill_value,
	long fill_value_idx
	)
{
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_fill", [&]() {
        AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, data.scalar_type(), "index_fill", [&]() {
	    index_fill<scalar_t, index_t>(
		at::cuda::getCurrentCUDAStream(),
		data.data_ptr<scalar_t>(),
		static_cast<index_t>(num_items),
		indices.data_ptr<index_t>(),
		counts.data_ptr<index_t>(),
		static_cast<index_t>(counts_idx),
		cat_outputs,
		fill_value.data_ptr<scalar_t>(),
		static_cast<index_t>(fill_value_idx)
		);
	});
    });
}

void pyt_index_fill_from(
	torch::Tensor data,
	long num_items,
	torch::Tensor indices,
	torch::Tensor counts,
	long counts_idx,
	bool cat_outputs,
	torch::Tensor fill_values
	)
{
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_fill_from", [&]() {
        AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, data.scalar_type(), "index_fill_from", [&]() {
	    index_fill_from<scalar_t, index_t>(
		at::cuda::getCurrentCUDAStream(),
		data.data_ptr<scalar_t>(),
		static_cast<index_t>(num_items),
		indices.data_ptr<index_t>(),
		counts.data_ptr<index_t>(),
		static_cast<index_t>(counts_idx),
		cat_outputs,
		fill_values.data_ptr<scalar_t>()
		);
	});
    });
}

} } }

