#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}
// use "__global__" to imply it's a kernel function
__global__ void SomeTransform(char *input_gpu, int fsize) {
	// find out the thread id(in x-dimension)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i=0; i<10; i++){
		int charIdx = idx*10 + i;
		if(charIdx < fsize and input_gpu[charIdx] != '\n' and input_gpu[charIdx] != ' ') {
			input_gpu[charIdx] = toupper(input_gpu[charIdx]);
		}
	}
	/*	
	if (idx < fsize and input_gpu[idx] != '\n') {
		input_gpu[idx] = '!';
	}
	*/
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	//format: kernel_name <<<gridDim,blockDim>>> (arg1, arg2, ...);
	int gridDim = 4;
	int blockDim = 32;
	SomeTransform<<<gridDim, blockDim>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());
	return 0;
}
