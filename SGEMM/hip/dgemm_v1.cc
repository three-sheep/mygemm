#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipblas.h>
#include <sys/time.h>


template<
    const int blockSize_sm,
    const int blockSize_sk,
    const int blockSize_sn,
    const int threadSize_gx,
    const int threadSize_gy
    >

__global__ void dgemm(const int M,const int K,const int N,double * __restrict__ A,
    double * __restrict__ B,double * __restrict__ C){
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // block中每个线程的id
    const int threads_num = (blockSize_sn/threadSize_gx)*(blockSize_sm/threadSize_gy); // 一个block中的线程数量
    //  threads_num = blockDim.x * blockDim.y;
    __shared__ double As[blockSize_sk*blockSize_sm];      // 一个线程块中存储的共享数据大小
    __shared__ double Bs[blockSize_sk*blockSize_sn];
    double lsg_a[blockSize_sk*blockSize_sm/threads_num];  // A写入SMEM中时进行的转置操作在寄存器写入数据时实现
    double fg_a[threadSize_gy]; // 读取一列数据放入寄存区中，列*行=矩。(As中的数据转置后，实际是读取的地址连续的一行)
    double fg_b[threadSize_gx];
    double accum[threadSize_gy*threadSize_gx]={0};  // 存放每个线程小迭代计算的矩阵结果

    // 每个线程块读取A矩阵数据时候的索引信息
    const int blockA_row_stride=threads_num/blockSize_sk; // 线程块一次拷贝多少行A矩阵的数据
    const int blockA_row = tid/blockSize_sk;  // 读取A矩阵时，每个线程对应的行信息
    const int blockA_col = tid%blockSize_sk;  // 因为传入的A矩阵是行主序列的一维数组，所以需要row,col信息来映射索引

    const int blockB_row_stride=threads_num/blockSize_sn;
    const int blockB_row = tid / blockSize_sn;
    const int blockB_col = tid %blockSize_sn;
    
    for(int iter1=0; iter1<K;iter1+=blockSize_sk){
        #pragma unroll
        // 从A【行主序】中读数据到As中，存入As时候用列主序
        for(int i=0;i<blockSize_sm;i+=blockA_row_stride){
            int lsg_index = i/blockA_row_stride; // 每个过渡寄存器中存的数据索引（行排列），连续地址追加即可
            lsg_a[lsg_index]=A[blockIdx.y*K*blockSize_sm +
                (blockA_row+i)*K + blockA_col + iter1];
            As[blockA_col*blockSize_sm+blockA_row+i]=lsg_a[lsg_index]; // 数据转置
        }
        // 读取数据时每个block,A为固定行滑动列读取；B为固定列滑动行读取。
        // A=&A[blockIdx.y*K*blockSize_sm]; B=&B[blockIdx.x*blockSize_sn]; 每个线程块的起始位置
        #pragma umroll
        for(int i=0;i<blockSize_sk;i+=blockB_row_stride){
            Bs[(blockB_row+i)*blockSize_sn+blockB_col]=B[blockIdx.x*blockSize_sn+
                (blockB_row+i+iter1)*N + blockB_col];
        }
        __syncthreads();  // 确保正确读取寄存器的值
        
        #pragma unroll
        for(int iter2=0; iter2<blockSize_sk; iter2++){
            #pragma unroll
            for(int a_c=0;a_c<threadSize_gy;a_c++){
                fg_a[a_c]=As[iter2*blockSize_sm+threadSize_gy * threadIdx.y + a_c];
            }
            #pragma unroll
            for(int b_c=0; b_c<threadSize_gx;b_c++){
                fg_b[b_c]=Bs[iter2*blockSize_sn+threadSize_gx * threadIdx.x + b_c];
            }
            __syncthreads();
            #pragma unroll
            for(int i=0;i<threadSize_gy;i++){
                for(int j=0;j<threadSize_gx;j++){
                    accum[i*threadSize_gx+j]+=fg_a[i]*fg_b[j];
                }
            }
            __syncthreads();
        }
    }

    // 将每个线程块计算出的结果合并到C
    #pragma unroll
    for(int i=0;i<threadSize_gy;i++){
        #pragma unroll
        for(int j=0;j<threadSize_gx;j++){
            C[(blockIdx.y * blockSize_sm + threadIdx.y * threadSize_gy + i) * N + 
                (blockIdx.x * blockSize_sn + threadIdx.x * threadSize_gx + j)]=accum[i*threadSize_gx+j];
        }
    }
}

int main(int argc, char * * argv){
    if(argc != 4){
        printf("usage: ./main [M] [N] [K]\n");
        exit(0);
    }
    size_t  M = atoi(argv[1]);
    size_t  K = atoi(argv[2]);
    size_t  N = atoi(argv[3]);

    //初始化 hipRAND 随机数发生器
    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    //计时器
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float dt;
    //分配矩阵内存空间
    double * A = (double * ) malloc(M * K * sizeof(double));
    double * B = (double * ) malloc(K * N * sizeof(double));
    double * C_kernel = (double * ) malloc(M * N * sizeof(double));
    double * C_hipblas= (double * ) malloc(M * N * sizeof(double));
    double * dA, * dB, *d_Ckernel, *d_Chipblas;
    hipMalloc(&dA, M * K * sizeof(double));
    hipMalloc(&dB, K * N * sizeof(double));
    hipMalloc(&d_Ckernel, M * N * sizeof(double));
    hipMalloc(&d_Chipblas, M * N * sizeof(double));
    //随机生成矩阵 A 和 B
    hipEventRecord(start);
    hiprandGenerateUniformDouble(gen, dA, M * K);
    hiprandGenerateUniformDouble(gen, dB, K * N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    printf("hiprand generating matrix: % 8.3f ms.\n", dt);
    //创建 hipBLAS 句柄
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;
    //预热
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
        &alpha, dB, N, dA, K, &beta, d_Chipblas, N);
    hipEventRecord(start);
    // blas为列主列存储，需要对输入参数做调整
    for(int i=0; i<1000;i++){
        hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
            &alpha, dB, N, dA, K, &beta, d_Chipblas, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    double FlopsR = (2*M*N*K)/dt;
    printf("hipblas api: %8.3f ms; Flops: %f \n", dt/1000,FlopsR);
    hipMemcpy(C_hipblas, d_Chipblas, N * N * sizeof(double),hipMemcpyDeviceToHost);

    const int blockSize_sm = 64;
    const int blockSize_sn = 64;
    const int blockSize_sk = 4;
    const int threadSize_gx = 4;
    const int threadSize_gy = 4;
    dim3 dimBlock(blockSize_sn/threadSize_gx,blockSize_sm/threadSize_gy);
    dim3 dimGrid(N/blockSize_sn,N/blockSize_sm);

    dt = 0.0;
    hipEventRecord(start);
    for(int i=0; i<1000;i++){
        dgemm<blockSize_sm,blockSize_sk,blockSize_sn,threadSize_gx,threadSize_gy>
            <<<dimGrid,dimBlock>>>(M,K,N,dA,dB,d_Ckernel);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    double FlopsC = (2*M*N*K)/dt;
    printf("self kernel: %8.3f ms; Flops: %f \n", dt/1000,FlopsC);
    
    hipMemcpy(C_kernel, d_Ckernel, N * N * sizeof(double),hipMemcpyDeviceToHost);
    //compare_matrices_result
    const double eps = 1.e-6;
    bool correct = true;
    for(int i=0; i<M*N; i++){
        double abs_err = fabs(C_kernel[i]-C_hipblas[i]);
        double rel_err = abs_err/(fabs(C_kernel[i]))/M;
        if(rel_err > eps){
            printf("Error ! Matrix[%d]=%.8f, ref=%.8f,error term is >%E\n",i,C_kernel[i],C_hipblas[i],eps);
            correct = false;
            break;
        }
    }
    printf("%s, ratio: %f. \n",correct ? "Result = PASS":"Result = FAIL",FlopsC/FlopsR);
    
    hipFree(dA);
    hipFree(dB);
    hipFree(d_Chipblas);
    hipFree(d_Ckernel);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hiprandDestroyGenerator(gen);
    hipblasDestroy(handle);
    free(A);
    free(B);
    free(C_hipblas);
    free(C_kernel);
    return 0;
}
