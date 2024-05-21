#include<stdlib.h>
#include<stdio.h>
#include"error.h"
#include<time.h>
#include<assert.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<
    const int blockSize_sm,
    const int blockSize_sk,
    const int blockSize_sn,
    const int threadSize_gx,
    const int threadSize_gy
    >

__global__ void sgemm(const int M,const int K,const int N,float * __restrict__ A,
    float * __restrict__ B,float * __restrict__ C){
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // thread index on each threads block
    const int threads_num = (blockSize_sn/threadSize_gx)*(blockSize_sm/threadSize_gy); 
    //  threads_num = blockDim.x * blockDim.y;
    __shared__ float As[2][blockSize_sk][blockSize_sm];    // double space used to alternate cache
    __shared__ float Bs[2][blockSize_sk][blockSize_sn];
    float lsg_a[blockSize_sk*blockSize_sm/threads_num]; // register buffer from global to shared
    float lsg_b[blockSize_sk*blockSize_sn/threads_num];
    float fg_a[2][threadSize_gy]; 
    float fg_b[2][threadSize_gx];
    float accum[threadSize_gy][threadSize_gx]={0};

    const int blockA_row_stride=threads_num/(blockSize_sk/4); 
    const int blockA_row = tid/(blockSize_sk/4);  
    const int blockA_col = tid%(blockSize_sk/4)*4;  

    const int blockB_row_stride=threads_num/(blockSize_sn/4);
    const int blockB_row = tid / (blockSize_sn/4);
    const int blockB_col = tid % (blockSize_sn/4)*4;
    
    // first title
    #pragma unroll
    for(int i=0; i<blockSize_sm; i+=blockA_row_stride){
        int lsg_index = i/blockA_row_stride*4; 
        FETCH_FLOAT4(lsg_a[lsg_index])=FETCH_FLOAT4(A[blockIdx.y*K*blockSize_sm +
                (blockA_row+i)*K + blockA_col]);
        for(int j=0;j<4;j++){
            As[0][blockA_col+j][blockA_row+i]=lsg_a[lsg_index+j]; 
        }
    }
    #pragma unroll
    for(int i=0;i<blockSize_sk;i+=blockB_row_stride){
        FETCH_FLOAT4(Bs[0][blockB_row+i][blockB_col])=FETCH_FLOAT4(B[blockIdx.x*blockSize_sn+
                (blockB_row+i)*N + blockB_col]);
        }
    __syncthreads();

    #pragma unroll
    for(int a_c=0;a_c<threadSize_gy;a_c+=4){
        FETCH_FLOAT4(fg_a[0][a_c])=FETCH_FLOAT4(As[0][0][threadSize_gy * threadIdx.y + a_c]);
    }
    #pragma unroll
    for(int b_c=0; b_c<threadSize_gx;b_c+=4){
        FETCH_FLOAT4(fg_b[0][b_c])=FETCH_FLOAT4(Bs[0][0][threadSize_gx * threadIdx.x + b_c]);
    }

    int write_stage = 1;
    for(int iter1=blockSize_sk; iter1<=K;iter1+=blockSize_sk){
        if (iter1<K){
            #pragma unroll
            for(int i=0;i<blockSize_sm;i+=blockA_row_stride){
                int lsg_index = i/blockA_row_stride*4; 
                FETCH_FLOAT4(lsg_a[lsg_index])=FETCH_FLOAT4(A[blockIdx.y*K*blockSize_sm +
                    (blockA_row+i)*K + blockA_col + iter1]);
            }
            #pragma umroll
            for(int i=0;i<blockSize_sk;i+=blockB_row_stride){
                int lsg_index = i/blockB_row_stride*4;
                FETCH_FLOAT4(lsg_b[lsg_index])=FETCH_FLOAT4(B[blockIdx.x*blockSize_sn+
                    (blockB_row+i+iter1)*N + blockB_col]);
            }
        }
        int read_stage = write_stage^1;

        #pragma unroll
        for(int iter2=0; iter2<blockSize_sk-1; iter2++){
            #pragma unroll
            for(int a_c=0;a_c<threadSize_gy;a_c+=4){
                FETCH_FLOAT4(fg_a[(iter2+1)%2][a_c])=FETCH_FLOAT4(As[read_stage][iter2+1][threadSize_gy * threadIdx.y + a_c]);
            }
            #pragma unroll
            for(int b_c=0; b_c<threadSize_gx;b_c+=4){
                FETCH_FLOAT4(fg_b[(iter2+1)%2][b_c])=FETCH_FLOAT4(Bs[read_stage][iter2+1][threadSize_gx * threadIdx.x + b_c]);
            }
            #pragma unroll
            for(int i=0;i<threadSize_gy;i++){
                for(int j=0;j<threadSize_gx;j++){
                    accum[i][j]+=fg_a[iter2%2][i]*fg_b[iter2%2][j];
                }
            }
        }
        if (iter1<K){
            #pragma unroll
            for(int i=0;i<blockSize_sm;i+=blockA_row_stride){
                int lsg_index = i/blockA_row_stride*4;
                for(int j=0;j<4;j++){
                    As[write_stage][blockA_col+j][blockA_row+i]=lsg_a[lsg_index+j]; 
                }
            }
            #pragma unroll
            for(int i=0;i<blockSize_sk;i+=blockB_row_stride){
                int lsg_index = i/blockB_row_stride*4;
                FETCH_FLOAT4(Bs[write_stage][blockB_row+i][blockB_col])=FETCH_FLOAT4(lsg_b[lsg_index]);
            }
            __syncthreads();
            write_stage ^=1;
        }
        // last iter2
        #pragma unroll
        for(int a_c=0;a_c<threadSize_gy;a_c+=4){
            FETCH_FLOAT4(fg_a[0][a_c])=FETCH_FLOAT4(As[read_stage^1][0][threadSize_gy * threadIdx.y + a_c]);
        }
        #pragma unroll
        for(int b_c=0; b_c<threadSize_gx;b_c+=4){
            FETCH_FLOAT4(fg_b[0][b_c])=FETCH_FLOAT4(Bs[read_stage^1][0][threadSize_gx * threadIdx.x + b_c]);
        }
        #pragma unroll
        for(int i=0;i<threadSize_gy;i++){
            for(int j=0;j<threadSize_gx;j++){
                accum[i][j]+=fg_a[1][i]*fg_b[1][j];
            }
        }
    }

    #pragma unroll
    for(int i=0;i<threadSize_gy;i++){
        #pragma unroll
        for(int j=0;j<threadSize_gx;j+=4){
            FETCH_FLOAT4(C[(blockIdx.y * blockSize_sm + threadIdx.y * threadSize_gy + i) * N + 
                (blockIdx.x * blockSize_sn + threadIdx.x * threadSize_gx + j)])=FETCH_FLOAT4(accum[i][j]);
        }
    }
}


int main(int argc, char** argv){
    if(argc != 4){
        printf("usage: ./main [M] [N] [K]\n");
        exit(0);
    }
    size_t  M = atoi(argv[1]);
    size_t  K = atoi(argv[2]);
    size_t  N = atoi(argv[3]);
    assert( M%8 == 0);      
    assert( N%8 == 0); 
    assert( K%8 == 0);

    // allocate memory sapace
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_Cc = (float*)malloc(bytes_C);
    float* h_Cr = (float*)malloc(bytes_C); 
    float* d_A;
    float* d_B;
    float* d_C;
    CheckCudaError(cudaMalloc(&d_A,bytes_A));
    CheckCudaError(cudaMalloc(&d_B, bytes_B));
    CheckCudaError(cudaMalloc(&d_C, bytes_C));

    // params
    const int blockSize_sm = 128;
    const int blockSize_sn = 128;
    const int blockSize_sk = 8;
    const int threadSize_gx = 8;
    const int threadSize_gy = 8;

    dim3 dimBlock( blockSize_sn/threadSize_gx,blockSize_sm/threadSize_gy);
    dim3 dimGrid(N/blockSize_sn,M/blockSize_sm);

    // generation random matrix 
    srand(time(NULL));
    for(int i=0; i<M*K; i++){h_A[i]=((float)rand()/RAND_MAX);}
    for(int i=0; i<N*K; i++){h_B[i]=((float)rand()/RAND_MAX);}

    CheckCudaError(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy( d_C, h_Cc, bytes_C, cudaMemcpyHostToDevice));

    float elapsed_time[2] = {0,0};
    cudaEvent_t start, stop;
    CheckCudaError(cudaEventCreate(&start));
    CheckCudaError(cudaEventCreate(&stop));
    
    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    CheckCudaError(cudaMemcpy( d_C, h_Cr, bytes_C, cudaMemcpyHostToDevice));
    // warmup
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    CheckCudaError(cudaEventRecord(start));
    for (int iter = 0 ; iter < 1000; iter++ ) {
        cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    }
    CheckCudaError(cudaEventRecord(stop));
    CheckCudaError(cudaEventSynchronize(stop));
    CheckCudaError(cudaEventElapsedTime(&elapsed_time[0], start, stop));
    CheckCudaError(cudaMemcpy( h_Cr, d_C, bytes_C, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    double FlopsR = (2*M*N*K)/(elapsed_time[0]);
    printf("cublas sgemm time:%f ;Flops: %f \n", elapsed_time[0]/1000,FlopsR);
    
    CheckCudaError(cudaMemset(d_C,0,bytes_C));
    CheckCudaError(cudaEventRecord(start));
    for(int iter = 0; iter<1000;iter++){
       sgemm<blockSize_sm,blockSize_sk,blockSize_sn,threadSize_gx,threadSize_gy>
            <<<dimGrid,dimBlock>>>(M,K,N,d_A,d_B,d_C);
    }
    CheckCudaError(cudaEventRecord(stop));
    CheckCudaError(cudaEventSynchronize(stop));
    CheckCudaError(cudaEventElapsedTime(&elapsed_time[1], start, stop));
    CheckCudaError(cudaMemcpy( h_Cc, d_C, bytes_C, cudaMemcpyDeviceToHost));
    double FlopsC = (2*M*N*K)/(elapsed_time[1]);
    printf("self_sgemm time:%f ;Flops: %f \n", elapsed_time[1]/1000,FlopsC);
    
    CheckResult(M,N,h_Cc,h_Cr);
    printf("ratio: %f\n",FlopsC/FlopsR);
    // 测试PR请求
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_Cc);
    free(h_Cr);
}

