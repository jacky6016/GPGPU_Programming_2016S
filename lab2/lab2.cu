#include "lab2.h"
#include <math.h>
#include <stdio.h>
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 960;
static const double freq = (double)1/(double)96;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

#define PI 3.1415926535

/*
	Using simplex noise 
	Reference: https://gist.github.com/Slipyx/2372043
*/

// Gradient
class Grad {
public:
    Grad( int8_t x, int8_t y, int8_t z ) : x(x), y(y), z(z) {}
    int8_t x, y, z;
};

/*
	reminder: a host variable cannot be directly read in a device function
	use a kernel(i.e __global__ function) to pass the data or declare it as a __device__ variable
*/
//this contains all the numbers between 0 and 255, these are put in a random order depending upon the seed
const uint8_t p[256] = { 151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 190, 6,148,247,
                                  120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 88,237,149,56,87,174,20,125,136,171,168, 68,175,74,
				                          165,71,134,139,48,27,166, 77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 102,143,54, 
				                          65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196, 135,130,116,188,159,86,164,100,109,198,173,186, 3,
				                          64,52,217,226,250,124,123, 5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 223,183,170,
				                          213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9, 129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 
				                          112,104,218,246,97,228, 251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,49,192,214, 31,181,
				                          199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

const Grad grad3[12] = {
    Grad(1,1,0),Grad(-1,1,0),Grad(1,-1,0),Grad(-1,-1,0),Grad(1,0,1),
    Grad(-1,0,1),Grad(1,0,-1),Grad(-1,0,-1),Grad(0,1,1),Grad(0,-1,1),
    Grad(0,1,-1),Grad(0,-1,-1)
};

uint8_t perm[512] = {0};
uint8_t permMod12[512] = {0};

// Initialize permutaion arrays
void init() {
    for ( uint16_t i = 0; i < 512; ++i ) {
        perm[i] = p[i & 255];
        permMod12[i] = static_cast<uint8_t>(perm[i] % 12);
    }
}

// Reminder:calling a __host__ function from a __global__ function is not allowed
// Fast floor
__device__ int32_t fastFloor( double x ) {
    int32_t xi = static_cast<int32_t>(x);
    return x < xi ? xi - 1 : xi;
}

__device__ double dot( const Grad& g, double x, double y ) {
    return g.x * x + g.y * y;
}

// 2D Simplex noise
__global__ void noise(uint8_t *imgptr, const Grad *grad3, uint8_t *perm, uint8_t *permMod12, int t, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		
	double xin = (double)(idx % W) / (double)W;
	double yin = (double)(idx / W) / (double)H;
	// Skewing/Unskewing factors for 2D
	const double F2 = 0.5 * (sqrt( 3.0 ) - 1.0);
	const double G2 = (3.0 - sqrt( 3.0 )) / 6.0;

	if(idx < size)
	{
		double s = (xin + yin) * F2;
    int32_t i = fastFloor( xin + s );
    int32_t j = fastFloor( yin + s );
    double t = (i + j) * G2;
    double x0 = xin - (i - t);
    double y0 = yin - (j - t);
    uint8_t i1 = 0, j1 = 1;
    if ( x0 > y0 ) {
        i1 = 1;
        j1 = 0;
    }
    double x1 = x0 - i1 + G2;
    double y1 = y0 - j1 + G2;
    double x2 = x0 - 1.0 + 2.0 * G2;
    double y2 = y0 - 1.0 + 2.0 * G2;
    uint8_t ii = i & 255;
    uint8_t jj = j & 255;
    uint8_t gi0 = permMod12[ii + perm[jj]];
    uint8_t gi1 = permMod12[ii + i1 + perm[jj + j1]];
    uint8_t gi2 = permMod12[ii + 1 + perm[jj + 1]];
    double n0 = 0.0;
    double t0 = 0.5 - x0 * x0 - y0 * y0;
    if ( t0 >= 0.0 ) {
        t0 *= t0;
        n0 = t0 * t0 * dot( grad3[gi0], x0, y0 );
    }
    double n1 = 0.0;
    double t1 = 0.5 - x1 * x1 - y1 * y1;
    if ( t1 >= 0.0 ) {
        t1 *= t1;
        n1 = t1 * t1 * dot( grad3[gi1], x1, y1 );
    }
    double n2 = 0.0;
    double t2 = 0.5 - x2 * x2 - y2 * y2;
    if ( t2 >= 0.0 ) {
        t2 *= t2;
        n2 = t2 * t2 * dot( grad3[gi2], x2, y2 );
    }
    imgptr[idx] = (int)(floor((70.0 * (n0 + n1 + n2) + 1)*128));
	}
	printf("test");
}

__global__ void trigonometric(uint8_t *imgptr,int t, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		
	double x = double(idx % W) / (double)W;
	double y = double(idx / W) / (double)H;
	if(idx < size) 
	{
		imgptr[idx] = (int)(floor((sin(2.0 * PI * x * 3 * t) + 1) * 128.0));		
	}
}


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
	// fps = 96/1 = 24
	info.fps_n = 96;
	info.fps_d = 1;
};

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
	int gridsize = H*W/blocksize + (H*W % blocksize == 0 ?0 :1);

	init();
	trigonometric<<<gridsize, blocksize>>>(imgptr, impl->t, H*W);
	cudaMemcpy(yuv, imgptr, H*W, cudaMemcpyDeviceToDevice); 
	cudaDeviceSynchronize();

	trigonometric<<<gridsize, blocksize>>>(imgptr+(H*W), impl->t, H*W/4);
	cudaMemcpy(yuv+(H*W), imgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	trigonometric<<<gridsize, blocksize>>>(imgptr+(H*W)+(H*W/4), impl->t, H*W/4);
	cudaMemcpy(yuv+(H*W)+(H*W)/4, imgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	++(impl->t);
	cudaFree(imgptr);
}
