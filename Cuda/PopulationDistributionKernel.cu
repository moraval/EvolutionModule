#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"


extern "C"
{
	// Kernel for updating the distribution for the outer evolution
	// Using Parzen Kernel distribution estimation
	__global__ void UpdateProbabilityDistribution(
		float* nrOfWeights,
		float* nrOfCoefficients,
		float* fitness,
		float* probabilities,
		float* notConverged,
		int distributionSize,
		int h)
	{
		int threadId = blockIdx.x*blockDim.x + threadIdx.x;
		float localProb = 0;
		float thisNrWeights = nrOfWeights[threadId];
		float thisNrCoeff = nrOfCoefficients[threadId];

		int i;
		float norm = (float)h*h;

		// copy all to shared memories - so there would be not problem with going to the same place in mem.
		extern __shared__ float cacheFit[];

		if (threadIdx.x < distributionSize){
			cacheFit[threadIdx.x] = fitness[threadIdx.x];
		}
		__syncthreads();

		if (threadId < distributionSize && notConverged[threadId] == 1)
		{
			for (i = 0; i < distributionSize; i++){

				// x(i) is # of coefficients & # of weights
				float gaussianKernel = __expf(-0.5*((i - threadId)*(i - threadId))/ norm);

				localProb += cacheFit[i] * (1 / norm) * gaussianKernel;
			}
		}

		__syncthreads();
		probabilities[threadId] = localProb;

		__syncthreads();

		float localSum;
		__shared__ float sum;

		if (threadIdx.x == 0){
			localSum = 0;
			for (i = 0; i < distributionSize; i++){
				localSum += probabilities[i];
			}
			sum = localSum;
		}
		__syncthreads();

		probabilities[threadId] = probabilities[threadId] / sum;

	}
}
