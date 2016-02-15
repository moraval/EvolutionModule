//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "ActivationFunction.cu"


extern "C"
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_OUTPUT_UNITS;
	__constant__ ActivationFunctionEnum D_ACTIVATION_FUNCTION;


	//edited code from Brainsimulator for computation of
	//output from the hidden layer of RNN (recurrent neural network)
	__global__ void ForwardPassHiddenKernel(
		float *input,
		float *hiddenActivations,
		float *previousHiddenActivations,
		float *inputWeights,
		float *recurrentWeights,
		int hiddenLayerSize
		)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		extern __shared__ float sharedMem[];
		if (threadIdx.x == 0){
			for (int i = 0; i < D_INPUT_UNITS; i++)
			{
				sharedMem[i] = input[i];
			}
			for (int i = D_INPUT_UNITS; i < D_INPUT_UNITS + hiddenLayerSize; i++)
			{
				sharedMem[i] = previousHiddenActivations[i-D_INPUT_UNITS];
			}
		}
		__syncthreads();

		if (unitId < hiddenLayerSize)
		{
			float weightedSum = 0;

			int weightId = unitId * D_INPUT_UNITS;
			for (int i = 0; i < D_INPUT_UNITS; i++)
			{
				weightedSum += inputWeights[weightId] * sharedMem[i];
				weightId++;
			}

			weightId = unitId * hiddenLayerSize;
			for (int i = 0; i < hiddenLayerSize; i++)
			{
				weightedSum += recurrentWeights[weightId] * sharedMem[i + D_INPUT_UNITS];
				weightId++;
			}

			hiddenActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}

	//edited code from Brainsimulator for computation of
	//output from the output layer of RNN (recurrent neural network)
	__global__ void ForwardPassOutputKernel(
		float *hiddenActivations,
		float *outputActivations,
		float *outputWeights,
		int hiddenLayerSize
		)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		extern __shared__ float sharedMem[];
		if (threadIdx.x == 0){
			for (int i = 0; i < hiddenLayerSize; i++)
			{
				sharedMem[i] = hiddenActivations[i];
			}
		}
		__syncthreads();

		if (unitId < D_OUTPUT_UNITS)
		{
			float weightedSum = 0;

			int weightId = unitId * hiddenLayerSize;
			for (int i = 0; i < hiddenLayerSize; i++)
			{
				weightedSum += outputWeights[weightId] * sharedMem[i];
				weightId++;
			}

			outputActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}
}
