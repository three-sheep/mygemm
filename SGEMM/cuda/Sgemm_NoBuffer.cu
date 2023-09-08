#include<stdlib.h>
#include<stdio.h>
#include"error.h"
#include<time.h>
#include<assert.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>

// 一次读取地址连续的4个float数，加快读取效率
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
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // block中每个线程的id
    const int threads_num = (blockSize_sn/threadSize_gx)*(blockSize_sm/threadSize_gy); // 一个block中的线程数量
    //  threads_num = blockDim.x * blockDim.y;
    __shared__ float As[blockSize_sk][blockSize_sm];      // 一个线程块中存储的共享数据大小
    __shared__ float Bs[blockSize_sk][blockSize_sn];
    float lsg_a[blockSize_sk*blockSize_sm/threads_num];  // A写入SMEM中时进行的转置操作在寄存器写入数据时实现
    float fg_a[threadSize_gy]; // 读取一列数据放入寄存区中，列*行=矩。(As中的数据转置后，实际是读取的地址连续的一行)
    float fg_b[threadSize_gx];
    float accum[threadSize_gy][threadSize_gx]={0};  // 存放每个线程小迭代计算的矩阵结果

    // 每个线程块读取A矩阵数据时候的索引信息
    const int blockA_row_stride=threads_num/(blockSize_sk/4); // 线程块一次拷贝多少行A矩阵的数据
    const int blockA_row = tid/(blockSize_sk/4);  // 读取A矩阵时，每个线程对应的行信息
    const int blockA_col = tid%(blockSize_sk/4)*4;  // 因为传入的A矩阵是行主序列的一维数组，所以需要row,col信息来映射索引

    const int blockB_row_stride=threads_num/(blockSize_sn/4);
    const int blockB_row = tid / (blockSize_sn/4);
    const int blockB_col = tid % (blockSize_sn/4)*4;
    
    // 没有使用双缓冲的预读取时，累加K/b_k次大迭代中每个线程块计算的结果即得到C的全部值，大迭代中包含b_k次更小的向量矩阵乘
    for(int iter1=0; iter1<K;iter1+=blockSize_sk){
        #pragma unroll
        // 从A【行主序】中读数据到As中，存入As时候用列主序
        for(int i=0;i<blockSize_sm;i+=blockA_row_stride){
            int lsg_index = i/blockA_row_stride*4; // 每个过渡寄存器中存的数据索引（行排列），连续地址追加即可
            FETCH_FLOAT4(lsg_a[lsg_index])=FETCH_FLOAT4(A[blockIdx.y*K*blockSize_sm +
                (blockA_row+i)*K + blockA_col + iter1]);
            for(int j=0;j<4;j++){
                As[blockA_col+j][blockA_row+i]=lsg_a[lsg_index+j]; // 数据转置，一个个按列追加写入SMEM中
                // As[blockA_col][blockA_row + i]=lsg_a[lsg_index];   
                // As[blockA_col+1][blockA_row + i]=lsg_a[lsg_index+1];
                // As[blockA_col+2][blockA_row + i]=lsg_a[lsg_index+2];
                // As[blockA_col+3][blockA_row + i]=lsg_a[lsg_index+3];
            }
        }
        // 读取数据时每个block,A为固定行滑动列读取；B为固定列滑动行读取。
        // A=&A[blockIdx.y*K*blockSize_sm]; B=&B[blockIdx.x*blockSize_sn]; 每个线程块的起始位置
        #pragma umroll
        for(int i=0;i<blockSize_sk;i+=blockB_row_stride){
            FETCH_FLOAT4(Bs[blockB_row+i][blockB_col])=FETCH_FLOAT4(B[blockIdx.x*blockSize_sn+
                (blockB_row+i+iter1)*N + blockB_col]);
        }
        __syncthreads();  // 确保正确读取寄存器的值
        
        // 小迭代，计算共享内存中的小矩阵数据。取每一行的数据至寄存器中，加快线程计算时访问数据的速度
        #pragma unroll
        for(int iter2=0; iter2<blockSize_sk; iter2++){
            #pragma unroll
            for(int a_c=0;a_c<threadSize_gy;a_c+=4){
                FETCH_FLOAT4(fg_a[a_c])=FETCH_FLOAT4(As[iter2][threadSize_gy * threadIdx.y + a_c]);
            }
            #pragma unroll
            for(int b_c=0; b_c<threadSize_gx;b_c+=4){
                FETCH_FLOAT4(fg_b[b_c])=FETCH_FLOAT4(Bs[iter2][threadSize_gx * threadIdx.x + b_c]);
            }
            __syncthreads();
            

            #pragma unroll
            for(int i=0;i<threadSize_gy;i++){
                for(int j=0;j<threadSize_gx;j++){
                    accum[i][j]+=fg_a[i]*fg_b[j];
                }
            }
            __syncthreads();
        }
    }

    // 将每个线程块计算出的结果合并到C
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

    dim3 dimBlock(blockSize_sn/threadSize_gx,blockSize_sm/threadSize_gy);
    dim3 dimGrid(N/blockSize_sn,M/blockSize_sm);

    // 模拟矩阵，以一维数组形式存储
    srand(10);
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

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_Cc);
    free(h_Cr);
}

