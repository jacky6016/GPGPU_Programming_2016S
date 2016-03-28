#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_one
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x == 1;
  }
};

// Some help kernel functions for accomplishing required tasks
__global__ void par_init(const char *text, int text_size, int *pos, int *lastpos)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < text_size) {
		if(text[idx] == '\n')
			pos[idx] = lastpos[idx] = 0;
		else
			pos[idx] = lastpos[idx] = 1;
	}
}

__global__ void cal_position(int *pos, int *lastpos, int text_size, int i)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < text_size && idx-i <= 0) {
		if(lastpos[idx] != 0 && lastpos[idx] == lastpos[idx-1])
			pos[idx] += lastpos[idx-i];
	}	
}

__global__ void update(int *pos, int *lastpos, int text_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
			
	if(idx < text_size) 
		lastpos[idx] = pos[idx];
}

__global__ void set_char_mask(const char *text, int text_size, char ch, int *output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < text_size) {
		if(text[idx] == ch)
			output[idx] = 1;
		else
		output[idx] = 0;
	}
}


void CountPosition(const char *text, int *pos, int text_size)
{
	/* Write the letter positions in each word(separated by '\n') of *text into *pos */

	int blocksize = 512; // generally can't run more than 512 threads in a block
	int gridsize = text_size / blocksize + (text_size % blocksize == 0 ?0 : 1); // one thread per character
	int *lastpos; // array storing the last result for referencing 
	size_t arraysize = sizeof(int) * text_size;	
	cudaMalloc((void **)&lastpos, arraysize);

	par_init<<<gridsize, blocksize>>>(text, text_size, pos, lastpos);
	// use this to synchronize kernel threads to prevent data hazards
	cudaDeviceSynchronize();
	// look back to as far as 2^9 = 512 characters(longest word length = 500) behind the current one
	for(int i=0; i<10; i++) { 
		int lookback_dist = 1 << i;
		cal_position<<<gridsize, blocksize>>>(pos, lastpos, text_size, lookback_dist);
		cudaDeviceSynchronize();
		// skip the last unnecessary update
		if(i == 9) break;
		update<<<gridsize, blocksize>>>(pos, lastpos, text_size);
		cudaDeviceSynchronize();
	}
	cudaFree(lastpos);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	/* Count */	
	// Use thrust functions to accomplish the tasks in this part.
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	// wrap raw pointers into device pointers to use thrust functions
	// thrust::device_ptr<type> wrap_ptr_name(raw_ptr_name);
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	nhead = thrust::count(pos_d, pos_d+text_size, 1);

	thrust::equal_to<int> op_equal;
	// Use the flags as a mask to find the starting position of words
	thrust::fill(cumsum_d, cumsum_d+text_size, 1);  			
	thrust::transform(pos_d, pos_d+text_size, cumsum_d,flag_d,op_equal);
	thrust::sequence(cumsum_d, cumsum_d+text_size);
	thrust::copy_if(cumsum_d, cumsum_d+text_size, flag_d, head_d, is_one());  // write the result to head_d
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{	
	// find the occurences of the vowels
	int numA, numE, numI, numO, numU;
	int *buff;
	int blocksize = 512;
	int gridsize = text_size / blocksize + (text_size % blocksize == 0 ?0 : 1); // one thread per character
	size_t arraysize = sizeof(int) * text_size;	
	cudaMalloc((void **)&buff, arraysize);

	thrust::device_ptr<int> buff_d(buff);
	
	set_char_mask<<<gridsize, blocksize>>>(text, text_size, 'A', buff);
	cudaDeviceSynchronize();
	numA = thrust::count(buff_d, buff_d+text_size, 1);

	set_char_mask<<<gridsize, blocksize>>>(text, text_size, 'E', buff);
	cudaDeviceSynchronize();
	numE = thrust::count(buff_d, buff_d+text_size, 1);

	set_char_mask<<<gridsize, blocksize>>>(text, text_size, 'I', buff);
	cudaDeviceSynchronize();
	numI = thrust::count(buff_d, buff_d+text_size, 1);

	set_char_mask<<<gridsize, blocksize>>>(text, text_size, 'O', buff);
	cudaDeviceSynchronize();
	numO = thrust::count(buff_d, buff_d+text_size, 1);

	set_char_mask<<<gridsize, blocksize>>>(text, text_size, 'U', buff);
	cudaDeviceSynchronize();
	numU = thrust::count(buff_d, buff_d+text_size, 1);

	printf("frequency of A = %d, E = %d, I = %d, O = %d, U = %d\n", numA, numE, numI, numO, numU);
	cudaFree(buff);
}
