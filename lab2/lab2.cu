#include "lab2.h"
#include <math.h>

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

#define PI 3.1415926535

// Using simplex noise 

//this contains all the numbers between 0 and 255, these are put in a random order depending upon the seed
int perm[] = { 151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 190, 6,148,247,
                   120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 88,237,149,56,87,174,20,125,136,171,168, 68,175,74,
				   165,71,134,139,48,27,166, 77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 102,143,54, 
				   65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196, 135,130,116,188,159,86,164,100,109,198,173,186, 3,
				   64,52,217,226,250,124,123, 5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 223,183,170,
				   213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9, 129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 
				   112,104,218,246,97,228, 251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,49,192,214, 31,181,
				   199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};


Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

/* 
	This function is called first to get the following info of the video:
	Height H,
	Width W,
	FPS(frames per second) = N/D,
	Number of frames Nf
*/
void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 48;
	info.fps_d = 1;
};

__global__ void trigonometric(uint8_t *imgptr,int t, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		
	if(idx < size) 
	{
		imgptr[idx] = (int)(floor((sin(idx % W * t * 2.0 * PI) + 1) * 128.0));
	}
}

/*
	This function is called Nf times at the rate of FPS
	Modify this function to get desired frames
*/
void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	/* 
		gray scale example: from RGB #000000(black) to #FFFFFF(white)
		write the Y channel with a value increased by time, then write the U & V channels(both have size W*H/4) with value 128
		So we get R=G=B=Y with an increasing Y
	*/
/*	
	cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
*/
	// modify the content of the memory this pointer points to
	// and copy it to  *yuv
	uint8_t * imgptr;
	cudaMalloc((void **) &imgptr, H*W*sizeof(uint8_t));

	int blocksize = 32;
	int gridsize = H*W/blocksize + (H*W % blocksiz == 0 ?0 :1);

	trigonometric<<<gridsize, blocksize>>>(imgptr, t, H*W);
	cudaMemcpy(yuv, imgptr, H*W, cudaMemcpyDeviceToDevice); 
	cudaDeviceSynchronize();

	trigonometric<<<gridsize, blocksize>>>(imgptr+(H*W), t, H*W/4);
	cudaMemcpy(yuv+(H*W), imgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	trigonometric<<<gridsize, blocksize>>>(imgptr+(H*W)+(H*W/4), t, H*W/4);
	cudaMemcpy(yuv+(H*W)+(H*W)/4, imgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	++(impl->t);
	cudaFree(imgptr);
}
