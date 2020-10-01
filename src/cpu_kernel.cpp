//*************************************************
////Author: Xianglong Kong
////Email: dinosaur8312@hotmail.com
////All rights reserved.
////*************************************************
//
#include "covmatrix.h"

void cov_matrix_cpu(complex<float> *A, complex<float> *covA, int *nx)
{
    int m=nx[0];    //sample numer
    int n=nx[1];    //feature number
    int l=nx[2];

    // mean calculation
    complex<float>*meanA = (complex<float>*) malloc(n*l*sizeof(complex<float>));
    for(int k=0;k<l;k++)
    {
        for(int j=0;j<n;j++)
        {
            int offset = k*n+j;
            complex<double> myMean = 0; //use double to avoid round-off error
            for(int i=0;i<m;i++)
            {
                int id=offset*m+i;
                myMean+=A[id];   
            }
            meanA[offset] = myMean/(double)m;
        }
    }

    //subtract all elements by mean
    complex<float>*As = (complex<float>*) malloc(m*n*l*sizeof(complex<float>));
    memcpy(As,A,m*n*l*sizeof( complex<float>));
    for(int k=0;k<l;k++)
    {
        for(int j=0;j<n;j++)
        {
            int offset = k*n+j;
            for(int i=0;i<m;i++)
            {
                int id=offset*m+i;
                As[id]=A[id]-meanA[offset];   
            }
        }
    }

    //covariance calculation
    for(int k=0;k<l;k++)
    {
        for(int i1=0;i1<n;i1++)
        {
            for(int i2=0;i2<n;i2++)
            {
                complex<float> mySum=0;
                for(int j=0;j<m;j++)
                {
                    int id1=k*n*m+i1*m+j;
                    int id2=k*n*m+i2*m+j;
                    mySum+=As[id1]*conj(As[id2]);
                }
                int id=k*n*n+i1*n+i2;
                covA[id]=mySum;
            }
        }
    }

    free(meanA);
    free(As);

}
