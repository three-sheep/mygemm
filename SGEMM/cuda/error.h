#pragma once
#include <stdio.h>

#define CheckCudaError(call)     \
{                       \
    const cudaError_t error_code = call;   \
    if (error_code != cudaSuccess)     \
    {                                  \
        printf("CUDA ERROR:\n");       \
        printf(" File: %s ,Line: %d ,Error code: %d \n", __FILE__,__LINE__,error_code); \
        printf("Error text: %s \n",cudaGetErrorString(error_code));         \
        exit(1);          \
    }                     \
}

// auto
void CheckResult(const int M,const int N,const float *A_c,const float *A_r){
    const double eps = 1.e-6;
    bool correct = true;
    for(int i=0; i<M*N; i++){
        int row = i/N;
        int col = i%N;
        double abs_err = fabs(A_c[i]-A_r[col*M + row]);
        double rel_err = abs_err/(fabs(A_c[i]))/M;
        if(rel_err > eps){
            printf("Error ! Matrix[%d]=%.8f, ref=%.8f,error term is >%E\n",i,A_c[i],A_r[col*M+row],eps);
            correct = false;
            break;
        }
    }
    printf("%s\n",correct ? "Result = PASS":"Result = FAIL");
}
