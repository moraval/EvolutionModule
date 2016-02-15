#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"
#include "math_constants.h"


extern "C"
{
	//The kernel for discrite cosine transformation
	__global__ void DiscriteCosineTransform(
		float* coefficients,
		float* output,
		float* weights,
		int C, //number of coefficients
		int W //number of weights
		)
	{
		int threadId = blockIdx.x*blockDim.x
			+ threadIdx.x;
		
		int w = threadId / C; //weight index
		int c = threadId % C; //coefficient index

		int offset = w*C;

		extern	__shared__ float cacheCoeff[];

		if (threadIdx.x < C){
			cacheCoeff[threadIdx.x] = coefficients[threadIdx.x];
		}
		
		__syncthreads();

		if (threadId < C*W){

			int isNotAtZero = c != 0;

			float s1 = CUDART_PI_F * c * w / C;

			float s2 = CUDART_PI_F * c / (C * 2);

			float local = isNotAtZero * __cosf(s1 + s2)
				+ !isNotAtZero * 1.f / sqrt(2.f);

			output[threadId] = cacheCoeff[c] * local;

		}
		__syncthreads();

		if (c == 0){
			float local = 0;
			for (int i = offset; i < C + offset; i++)
			{
				local += output[i];
				//weights[w] += output[i];
			}
			//local *= sqrt(2.f) / sqrt((float)C);
			weights[w] = local * sqrt(2.f) / sqrt((float)C);
		}
	}
}
