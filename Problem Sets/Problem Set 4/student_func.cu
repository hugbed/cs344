//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define THREADS_PER_BLOCK 512

typedef unsigned int uint;

__global__
void bitIsZeroPredicate(uint* d_oTruePredicate,
						uint* d_oFalsePredicate,
						uint* const d_iData,
						const uint offsetFromMSB,
						const size_t n)
{
	const uint id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (id >= n) {
		return;
	}
	
	uint predicate = static_cast<uint>((d_iData[id] & 1 << offsetFromMSB) == 0);
	
	d_oTruePredicate[id] = predicate;
	d_oFalsePredicate[id] = (1 - predicate);
}


__global__ void partialPreScan(uint *d_odata, uint *d_oblockSums, const uint *d_idata, uint n)
{
	/*
	 * For large arrays, do this operation with multiple blocks then
	 * do it again with one block with the blockSums to scan the blockSums
	 * call addScannedBlockSums with the scanned blockSums to merge the blocks.
	 * 
	 * Each block must have blockDim.x * sizeof(uint) as shared memory
	 */
	
	 extern __shared__ uint s_temp[];// allocated on invocation
	 
	 // copy into shared memory
	 const uint tid = threadIdx.x;
	 const uint id = blockIdx.x * blockDim.x + tid;
	  
     // pad the smaller block
	 if (id < n) {
		 s_temp[tid] = d_idata[id];
	 } else {
		 s_temp[tid] = 0;
	 }
	 __syncthreads();
	 
	 // upsweap
	 int offset = 1;
	 for (; offset < blockDim.x; offset <<= 1) {
		 if ((tid + 1) % (offset << 1) == 0) {
			 s_temp[tid] += s_temp[tid - offset];
		 }
		 __syncthreads();
	 }

	 // reset last value to identity element
	 if (tid == (blockDim.x - 1)) {
		 d_oblockSums[blockIdx.x] = s_temp[tid];
		 s_temp[tid] = 0;
	 }
	 __syncthreads();
	 
	 // downsweap
	 for (;offset > 0; offset >>= 1) {
		 if ((tid + 1) % (offset << 1) == 0) {
			 uint old = s_temp[tid - offset];
			 s_temp[tid - offset] = s_temp[tid];
			 s_temp[tid] += old;
		 }
		 __syncthreads();
	 }
	 
	 // copy result to global memory
	 if (id < n) {
		 d_odata[id] = s_temp[tid];
	 }
}

__global__ void addScannedBlockSums(uint* d_data, uint* d_isums, const size_t n)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		d_data[id] += d_isums[blockIdx.x]; 
	}
}

void preScan(uint *d_odata, uint *d_osum, const uint *d_idata, const uint N)
{
	const uint nbBlocks = ceil(float(N)/float(THREADS_PER_BLOCK));

	uint *d_osums;
	checkCudaErrors(cudaMalloc((void**)&d_osums, nbBlocks * sizeof(uint)));
	
	// Scan with multiple independant blocks
	checkCudaErrors(cudaMemset(d_osums, 0, nbBlocks*sizeof(unsigned int)));
	partialPreScan<<<nbBlocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(uint)>>>(d_odata, d_osums, d_idata, N);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// Scan the sum of these blocks
	partialPreScan<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(uint)>>>(d_osums, d_osum, d_osums, nbBlocks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	uint *h_osums = new uint[nbBlocks];
	checkCudaErrors(cudaMemcpy(h_osums, d_osums, nbBlocks*sizeof(uint), cudaMemcpyDeviceToHost));
	
	// Add the block sums to the other independant blocks
	addScannedBlockSums<<<nbBlocks, THREADS_PER_BLOCK>>>(d_odata, d_osums, N);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	checkCudaErrors(cudaFree(d_osums));
}

__global__ void scatter(uint* d_odata, const uint* d_idata,
						const uint* d_zeroScan, const uint* d_oneScan,
						const uint* d_zeroPredicate,
						const uint* numZeros)
{
	const uint id = blockDim.x * blockIdx.x + threadIdx.x;
	
	uint newAddr;
	if (d_zeroPredicate[id] == 1) {
		newAddr = d_zeroScan[id];
	} else {
		newAddr = d_oneScan[id] + (*numZeros);
	}
	
	d_odata[newAddr] = d_idata[id];
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	/*
	* radix_sort(in):
	* 	  for each bit:
	* 	   	  1) numZeros
	* 		  2) Predicate : 
	* 	  		  p_0[i] = (val[i] & bit) == 0
	* 			  p_1[i] = ~p_0
	* 		  3) Exclusive sum-scan for p_0 and p_1
	* 	       	  -->presum_0 = ex_sum_scan(p_0)	(gives adresses to put sorted in into out)
	* 	       	  -->presum_1 = ex_sum_scan(p_1)
	* 	       	      -->presum_1[i] += numZeros
	* 	      4) out <-- compact(in, presum_0)		(check for bounds)
	* 	         out <-- compact(in, presum_1)
	*/
//	initPreScan(numElems);
	
	uint *d_isZeroPredicate, *d_isOnePredicate;
	checkCudaErrors(cudaMalloc(&d_isZeroPredicate, numElems*sizeof(uint)));
	checkCudaErrors(cudaMalloc(&d_isOnePredicate, numElems*sizeof(uint)));
	
	uint *d_isZeroScan, *d_isOneScan;
	checkCudaErrors(cudaMalloc((void **)&d_isZeroScan, numElems*sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&d_isOneScan, numElems*sizeof(uint)));
	
	uint * d_nbZeros, * d_nbOnes;
	checkCudaErrors(cudaMalloc((void **)&d_nbZeros, sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&d_nbOnes, sizeof(uint)));
	
	const uint NB_BITS = 32;
	
	uint bitOffset = 0;
//	for (uint bitOffset = 0; bitOffset < NB_BITS; bitOffset++) {
		
	// compute predicate values
	uint numThreadsPerBlock = 256;
	uint numBlocks = numElems/numThreadsPerBlock + 1;
	bitIsZeroPredicate<<<numBlocks, numThreadsPerBlock>>>(
			d_isZeroPredicate, 
			d_isOnePredicate, 
			d_inputVals, 
			bitOffset,
			numElems);
	
	// compute pre scan of both predicates
	preScan(d_isZeroScan, d_nbZeros, d_isZeroPredicate, numElems);
	preScan(d_isOneScan, d_nbOnes, d_isOnePredicate, numElems);

	uint * h_nbZeros = new uint[numElems];
	uint * h_nbOnes = new uint[numElems];
	checkCudaErrors(cudaMemcpy(h_nbZeros, d_nbZeros, sizeof(uint), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_nbOnes, d_nbOnes, sizeof(uint), cudaMemcpyDeviceToHost));
	
	printf("nbZeros: %d, %d\n", *h_nbZeros, *h_nbOnes);
	
//	// put the sorted result at its correct index
//	// flip input/output each turn
//	if ((bitOffset + 1) % 2 == 1) {
//		scatter<<<gridSize, blockSize>>>(
//				d_inputVals, d_outputVals,
//				d_isZeroScan, d_isOneScan,
//				d_isZeroPredicate,
//				nbZeros);
//		scatter<<<gridSize, blockSize>>>(
//				d_inputPos, d_outputPos,
//				d_isZeroScan, d_isOneScan,
//				d_isZeroPredicate,
//				nbZeros);	
//	} else {
//		scatter<<<gridSize, blockSize>>>(
//				d_outputVals, d_inputVals,
//				d_isZeroScan, d_isOneScan,
//				d_isZeroPredicate,
//				nbZeros);
//		scatter<<<gridSize, blockSize>>>(
//				d_outputPos, d_inputPos,
//				d_isZeroScan, d_isOneScan,
//				d_isZeroPredicate,
//				nbZeros);
//	}
//	}
	
//	cleanPreScan();
			
	checkCudaErrors(cudaFree(d_isZeroPredicate));
	checkCudaErrors(cudaFree(d_isOnePredicate));
	checkCudaErrors(cudaFree(d_isZeroScan));
	checkCudaErrors(cudaFree(d_isOneScan));
}
