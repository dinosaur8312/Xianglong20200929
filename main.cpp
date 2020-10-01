//*************************************************
//Author: Xianglong Kong
//Email: dinosaur8312@hotmail.com
//All rights reserved.
//*************************************************

// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "covmatrix.h"
#include "gpu_kernel.cuh"
#include "cpu_kernel.h"

bool compare_results(complex<float> *cpu, complex<float>*gpu, int *nx)
{
    printf("Checking computed result for correctness...\n");
    bool correct = true;

    float eps = 1.E-5;  

    double norm_cpu=0;
    double norm_gpu=0;
    double norm_diff=0;

    for(int k=0;k<nx[2];k++)
    {
        for(int j=0;j<nx[1];j++)
        {
            for(int i=0;i<nx[1];i++)
            {
                int id = k*nx[1]*nx[1]+j*nx[1]+i;
                complex<float> diff = cpu[id]-gpu[id];
                if(abs(cpu[id])>eps) 
                {
                    float rel_err = abs(diff)/abs(cpu[id]);

                    if(rel_err>eps)
                    {
                        printf("Mismatch fount at [%d,%d,%d] \n",k,j,i);
                        correct = false;
                    }
            
                }
                norm_cpu+=norm(cpu[id]);
                norm_gpu+=norm(gpu[id]);
                norm_diff+=norm(diff);
            }
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    if(correct)
    {
        //printf("Relative error is smaller than machine epsilon for every element\n");
        norm_cpu=sqrtf(norm_cpu);
        norm_gpu=sqrtf(norm_gpu);
        norm_diff=sqrtf(norm_diff);
        //int numElem=nx
        printf("L2 Norm of CPU covariance matrix:        %8.4E\n",norm_cpu);
        printf("L2 Norm of CPU covariance matrix:        %8.4E\n",norm_gpu);
        printf("L2 Norm of covariance matrix difference: %8.4E\n",norm_diff);
        printf(" \n");
    }
}

void init_matrix(complex<float> *A, int *nx)
{
    
    for(int k=0;k<nx[2];k++)
    {
        for(int j=0;j<nx[1];j++)
        {
            for(int i=0;i<nx[0];i++)
            {
                int id = (k*nx[0]*nx[1]+j*nx[0]+i);
                float realp=cosf(0.1*i+0.3*j);
                float imagp =sinf(0.2*j+0.5*i);
                A[id]=(realp,imagp);
            }
        }
    }
}
void cov_matrix_test(int *nx)
{
    
    complex<float> *A=(complex<float> *)malloc(sizeof(complex<float>)*nx[0]*nx[1]*nx[2]);
    complex<float> *covA_gpu=(complex<float> *)malloc(sizeof(complex<float>)*nx[1]*nx[1]*nx[2]);
    complex<float> *covA_cpu=(complex<float> *)malloc(sizeof(complex<float>)*nx[1]*nx[1]*nx[2]);

    init_matrix(A, nx);       

    cov_matrix_gpu(A, covA_gpu, nx);

    cov_matrix_cpu(A, covA_cpu, nx);

    compare_results(covA_cpu, covA_gpu, nx);

    free(A);
    free(covA_cpu);
    free(covA_gpu);
    return;
}

int main() 
{
    init_gpu();

    for(int l=2048;l<10000;l*=2) //l=2048,4096,8192
    {
        for(int m=256; m<1000;m+=256) //m=256,512,768
        {
            for(int n=8; n<=16;n+=8) //n=8,16
            {
                printf("\n----------------------------------------------------\n");
                printf("Test case: (n,m,l)=(%d,%d,%d)\n",n,m,l);
                int nx[3] = {m,n,l};
        
                cov_matrix_test(nx);
            }
        }
    }
 
    return 0;               
}        

