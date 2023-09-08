#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipblas.h>
#include <sys/time.h>

struct my_timer
{
    timeval ts, te; //起始时刻,终止时刻
    float dt; // 时间间隔,单位毫秒(ms)
    void start(){
        gettimeofday(&ts, NULL);
    }
    void stop(){
        gettimeofday(&te, NULL);
        long int dt_sec = te.tv_sec-ts.tv_sec;
        long int dt_usec = te.tv_usec-ts.tv_usec;
        // 时间间隔是ms
        dt = dt_sec * 1.0e3 + dt_usec / 1.0e3;
    }
};

// 针对方阵的代码,线程块大小是16*16
#define Nsub 16
// 从M矩阵的第i行j列开始读取
__device__ void read_matrix(double* M, int N, int i, int j,double* M_ij, int LD){
    //矩阵元在子矩阵 M_ij 中是 row 行 col 列
    int row = threadIdx.x;
    int col = threadIdx.y;
    //矩阵元在 M_ij 中的偏移
    int offset = row + col * LD;
    //矩阵元在 M 中是 Row 行 Col 列
    int Row = i * Nsub + row;
    int Col = j * Nsub + col;
    //矩阵元在 M 中的偏移
    int Offset = Row + Col * N;
    //没有越界则读取 否则置零
    if ((Row < N) && (Col < N)) {
        M_ij[offset] = M[Offset];
    } 
    else {
        M_ij[offset] = 0.0;
    }
}


__global__ void matrix_multiply(int N, int Nblk, double * A, double * B,double * C){
    //声明 A_sub,B_sub 用于保存 A,B 的子矩阵1
    __shared__ double A_sub[Nsub * Nsub];
    __shared__ double B_sub[Nsub * Nsub];
    //每个线程块负责计算子矩阵 C_ij
    for (int i = blockIdx.x; i < Nblk; i += gridDim.x){
        for (int j = blockIdx.y; j < Nblk; j += gridDim.y) {
            //每个线程负责计算子矩阵 C_ij 中 row 行 col 列的矩阵元
            int row = threadIdx.x;
            int col = threadIdx.y;
            double C_sub = 0.0;
            //每个线程块需要迭代Nblk次
            for (int k = 0; k < Nblk; k++) {
            //将子矩阵 A_ik 从全局内存读到共享内存
                read_matrix(A, N, i, k, A_sub, Nsub);
                //将子矩阵 B_kj 从全局内存读到共享内存
                read_matrix(B, N, k, j, B_sub, Nsub);
            //计算矩阵 A_ik 和 B_kj 的乘
                __syncthreads();  //一个线程块多个线程束时，需要等待全部的线程运算结束
                for (int l = 0; l < Nsub; l++) {
                    C_sub += A_sub[row + l * Nsub]* B_sub[l + col * Nsub];
                }
                __syncthreads();
            }// k
            //将矩阵元写到 C 中
            int Row = i * Nsub + row;
            int Col = j * Nsub + col;
            if ((Row < N) && (Col < N)) {
                C[Row + Col * N] = C_sub;
            }
        } // j
    } // i
}

void compare_matrices(int N, double* C1, double* C2)
{
    int i, j;
    double max_error = 0.0;
    bool fail=false;
    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
            double error = fabs(C1[i+j* N]- C2[i+j* N]);
            if (error > max_error) {
                max_error = error;fail=true;break;
            }
        }
        if (fail){
            printf("Max error: % .4e, index:%d(%d) \n", max_error,i+j* N,N*N);
            break;
        }
    }
    if (fail){
        printf("Max error: % .4e, index:%d(%d) \n", max_error,i+j* N,N*N);}
    else
        printf("passed! \n");
}


int main(int argc, char * * argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N < 2) {
        fprintf(stderr, "N must larger than 2.\n");
        return 1;
    }
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
    double * A = (double * ) malloc(N * N * sizeof(double));
    double * B = (double * ) malloc(N * N * sizeof(double));
    double * C_kernel = (double * ) malloc(N * N * sizeof(double));
    double * C_hipblas= (double * ) malloc(N * N * sizeof(double));
    double * dA, * dB, *d_Ckernel, *d_Chipblas;
    hipMalloc(&dA, N * N * sizeof(double));
    hipMalloc(&dB, N * N * sizeof(double));
    hipMalloc(&d_Ckernel, N * N * sizeof(double));
    hipMalloc(&d_Chipblas, N * N * sizeof(double));
    //随机生成矩阵 A 和 B
    hipEventRecord(start);
    hiprandGenerateUniformDouble(gen, dA, N * N);
    hiprandGenerateUniformDouble(gen, dB, N * N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    printf("Generating Matrices took % 8.3f ms.\n", dt);
    //创建 hipBLAS 句柄
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    //矩阵乘法,将结果存储在矩阵 C 中
    const double alpha = 1.0;
    const double beta = 0.0;
    //预热,lda\ldb\ldc不同矩阵的主维度(row)
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 2, 2, 2,
        &alpha, dA, N, dB, N, &beta, d_Chipblas, N);
    hipEventRecord(start);
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
        &alpha, dA, N, dB, N, &beta, d_Chipblas, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    printf("blas api: %8.3f ms.\n", dt);
    hipMemcpy(C_hipblas, d_Chipblas, N * N * sizeof(double),hipMemcpyDeviceToHost);

    //hipMemset(dC,0,N*N*sizeof(double));
    dim3 blockSize = dim3(Nsub,Nsub,1);
    int blockN = (N+Nsub-1)/Nsub;
    dim3 numBlock = dim3(blockN,blockN,1); 
    hipEventRecord(start);
    matrix_multiply<<<numBlock,blockSize>>>(N,blockN,dA,dB,d_Ckernel);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    printf("self kernel: %8.3f ms.\n", dt);
    hipMemcpy(C_kernel, d_Ckernel, N * N * sizeof(double),hipMemcpyDeviceToHost);
    compare_matrices(N,C_kernel,C_hipblas);
    
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

