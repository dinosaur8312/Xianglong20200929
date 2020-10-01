//*************************************************
//Author: Xianglong Kong
//Email: dinosaur8312@hotmail.com
//All rights reserved.
//*************************************************


// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include "myutil.h"
#include "covmatrix.h"
#include "helper_cuda.h"


#define BLOCK_SIZE_X 32


__device__ float2 compute_mean(const float2 __restrict__ *A, int m)
{
    unsigned int base = (blockIdx.x*blockDim.y+threadIdx.y)*m;

    float mySum_r =  0;
    float mySum_i =  0;

    // Reduce multiple elements per thread.  
    unsigned int tid =  threadIdx.x;
    while (tid < m)
    {
        float2 r_A = A[base+tid];
        mySum_r += r_A.x;
        mySum_i += r_A.y;
        tid += blockDim.x;
    }

    //Reduction across warp
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        mySum_r += __shfl_down_sync(0xFFFFFFFF, mySum_r, offset);
        mySum_i += __shfl_down_sync(0xFFFFFFFF, mySum_i, offset);
    }

    float myMean_r, myMean_i;
    if(threadIdx.x==0)
    {
        myMean_r = mySum_r/m;
        myMean_i = mySum_i/m;
    }

    //Broadcast of mean value across warp
    myMean_r = __shfl_sync(0xFFFFFFFF, myMean_r, 0);   
    myMean_i = __shfl_sync(0xFFFFFFFF, myMean_i, 0);   

    return make_float2(myMean_r, myMean_i);
}




template <int BLOCK_SIZE_Y>__global__ void cov_matrix_kernel(const float2 __restrict__ *A, float2 __restrict__ *covA, int m)
{

    float2 myMean = compute_mean(A,m);

    __shared__ float2 s_data[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    unsigned int base = (blockIdx.x*blockDim.y+threadIdx.y)*m;

    float2 r_cov=make_float2(0,0);

    unsigned int tid =  threadIdx.x;
    while(tid<m)
    {
        s_data[threadIdx.y][threadIdx.x] = A[base+tid]-myMean;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_Y; k++) 
        {
            int tx = threadIdx.x%BLOCK_SIZE_Y;
            int id = k+threadIdx.x/BLOCK_SIZE_Y*BLOCK_SIZE_Y;
            r_cov += conj_mul(s_data[threadIdx.y][id], s_data[tx][id]);
        }

        tid += blockDim.x;
        __syncthreads();
    }

    float r_cov_r=r_cov.x;
    float r_cov_i=r_cov.y;

#pragma unroll
    for (int offset = 16; offset >=BLOCK_SIZE_Y; offset /= 2)
    {
        r_cov_r += __shfl_down_sync(0xFFFFFFFF, r_cov_r, offset);
        r_cov_i += __shfl_down_sync(0xFFFFFFFF, r_cov_i, offset);
    }
    if(threadIdx.x<BLOCK_SIZE_Y)
    {
        unsigned int id = (blockIdx.x*blockDim.y*blockDim.y+threadIdx.y*blockDim.y+threadIdx.x);
        covA[id]=make_float2(r_cov_r,r_cov_i);
    }
    
}


void init_gpu()
{
    int dev = findCudaDevice(0,NULL);
}

//Calculate Covariance Matrix using GPU
void cov_matrix_gpu(complex<float> *A, complex<float> *covA, int *nx) 
{
    // Initialize device pointers.
    float *d_A, *d_covA;
    int m=nx[0];    //sample numer
    int n=nx[1];    //feature number
    int l=nx[2];    

    int numElem_A = n*m*l;
    int numElem_covA = n*n*l;
    // Allocate device memory.
    cudaMalloc( (void **) &d_A, numElem_A * sizeof(float2));
    cudaMalloc( (void **) &d_covA, numElem_covA * sizeof(float2));

    cudaMemcpy(d_A, A, numElem_A * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_covA, covA, numElem_covA * sizeof(float2), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);


    // Calculate blocksize and gridsize.
    dim3 threads(BLOCK_SIZE_X, n);
    dim3 blocks(l);

    // Launch CUDA kernel.
    if(n==16)
        cov_matrix_kernel<16><<<blocks, threads>>>((float2 *)d_A, (float2 *)d_covA, m);
    else if(n==8)
        cov_matrix_kernel<8><<<blocks, threads>>>((float2 *)d_A, (float2 *)d_covA, m);
    else
    {
        printf("Unsupported N value of %d\n",n);
        return;
    }

     // Record the stop event
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);


    cudaMemcpy(covA, d_covA, numElem_covA * sizeof(float2), cudaMemcpyDefault);
    cudaFree(d_covA);
    cudaFree(d_A);
    cudaDeviceSynchronize();

 
    double gigaSize = (2*numElem_A+numElem_covA)*sizeof(float2)*1.0e-9f;
    double bandwidth = gigaSize/(msecTotal/1000);
    printf("GPU Time: %.3f msec\n",msecTotal);
    printf("GPU Performance: %.2f GB/s\n",bandwidth);
}











