/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#define OP_MAX 3
#define OP_MIN 2

#define THREADBLOCK_SIZE 256

typedef unsigned int uint;

__global__ 
void reduce(float* d_out,
			const float* const d_in,
			const unsigned int op)
{
	extern __shared__ float stmp[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// load data into shared memory (one value per thread)
	int tid = threadIdx.x;
	stmp[tid] = d_in[i];
	__syncthreads();

	for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			if (op == OP_MAX) {
				stmp[tid] = max(stmp[tid], stmp[tid + s]);
			} else if (op == OP_MIN){
				stmp[tid] = min(stmp[tid], stmp[tid + s]);
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = stmp[0];
	}
}

__global__
void histogram(uint* d_bins,
			   const float* const d_in,
			   const float lumMin,
			   const float lumMax,
			   const size_t numBins)
{
	extern __shared__ float s_in[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// load data into shared memory (one value per thread)
	int tid = threadIdx.x;
	s_in[tid] = d_in[i];
	__syncthreads();

	// could initialize to 0 some way here
	// here <--

	// could build histogram in shared memory, then
	// merge it to global histogram

	float lumRange = lumMax - lumMin;
	unsigned int binIdx = (s_in[tid] - lumMin) / lumRange * numBins;

	// could probably be faster somehow else
	atomicAdd(&(d_bins[binIdx]), 1);
}


////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}


inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint size
)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);

    //Write back
    d_Dst[pos] = odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;

static uint factorRadix2(uint &log2L, uint L)
{
    if (!L)
    {
        log2L = 0;
        return 0;
    }
    else
    {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);

        return L;
    }
}

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

extern "C" size_t scanExclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    //Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    //Check all threadblocks to be fully packed with data
    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        arrayLength
    );

    return THREADBLOCK_SIZE;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	
	// 1)
	size_t n = numRows * numCols;
	int blockSize = 1024;
	int gridSize = n / blockSize;
	
	float *d_intermediate, *d_out;
	checkCudaErrors(cudaMalloc(&d_intermediate, n * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));
	
	float h_out;
	reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_logLuminance, OP_MAX);
	reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_out, d_intermediate, OP_MAX);
	checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	max_logLum = h_out;
	
	reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_logLuminance, OP_MIN);
	reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_out, d_intermediate, OP_MIN);
	checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	min_logLum = h_out;

	checkCudaErrors(cudaFree(d_intermediate));
	checkCudaErrors(cudaFree(d_out));

	uint *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, numBins * sizeof(uint)));
	checkCudaErrors(cudaMemset(d_bins, 0, numBins * sizeof(uint)));
	histogram<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_bins, d_logLuminance, min_logLum, max_logLum, numBins);

	scanExclusiveShort(d_cdf, d_bins, 1, numBins);
}
