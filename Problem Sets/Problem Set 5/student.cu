/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

typedef unsigned int uint;

#define VALS_PER_COARSE 256 // 2^8

__global__
void coarseBins(uint *oVals,
				const uint *iVals,
				const size_t numValsPerCoarse,
				const size_t numVals)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numVals) {
		return;
	}

	// Divide by the number of values per coarse bin
	// to get the coarse bin index
	oVals[i] = iVals[i] / numValsPerCoarse; // with iVals[i] >> 8 instead would be faster
}

__global__
void yourHisto(const uint* const vals, //INPUT
               uint* const histo,      //OUPUT
               const size_t numVals)
{

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	uint gridSize = 0;
	uint blockSize = 0;

	uint* d_coarseVals;
	checkCudaErrors(cudaMalloc((void**)&d_coarseVals, numElems*sizeof(uint)));
	coarseBins<<<gridSize, blockSize>>>(d_coarseVals, d_vals, VALS_PER_COARSE, numElems);

	uint *h_coarse = new uint[numElems];
	checkCudaErrors()

	printf("numElems: %d", numElems);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
