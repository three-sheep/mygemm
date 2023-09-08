#include <iostream>
#include <stdio.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include <sys/time.h>

#define NUM 256
#define WARMUP

#define SCALAR_ZERO (double)(0)
#define LDS_NUM_ELEMENTS 2048
#define LDS_OFFSET_B 512
#define LDS_OFFSET_BLK 1024
#define DEPTH 8
#define MT0I 64
#define MT1J 64

/* MAC's */
#define MAC(A,B,DST) DST += A*B
#define TYPE_MAC(MULA,MULB,DST) DST = MAC(MULA,MULB,DST);
#define TT 4
#define SIZE_HALF_A 32
#define SIZE_HALF_B 32
#define SIZE_HALF_C 32

/* 4x4 micro-tile */
#define MAC_4x4\
  TYPE_MAC(rA[0],rB[0],rC[0]); \
  TYPE_MAC(rA[0],rB[1],rC[1]); \
  TYPE_MAC(rA[0],rB[2],rC[2]); \
  TYPE_MAC(rA[0],rB[3],rC[3]); \
  TYPE_MAC(rA[1],rB[0],rC[4]); \
  TYPE_MAC(rA[1],rB[1],rC[5]); \
  TYPE_MAC(rA[1],rB[2],rC[6]); \
  TYPE_MAC(rA[1],rB[3],rC[7]); \
  TYPE_MAC(rA[2],rB[0],rC[8]); \
  TYPE_MAC(rA[2],rB[1],rC[9]); \
  TYPE_MAC(rA[2],rB[2],rC[10]); \
  TYPE_MAC(rA[2],rB[3],rC[11]); \
  TYPE_MAC(rA[3],rB[0],rC[12]); \
  TYPE_MAC(rA[3],rB[1],rC[13]); \
  TYPE_MAC(rA[3],rB[2],rC[14]); \
  TYPE_MAC(rA[3],rB[3],rC[15]); \
  
#define MAC_4x4_BLK\
  TYPE_MAC(rA[0+TT],rB[0+TT],rC[0]); \
  TYPE_MAC(rA[0+TT],rB[1+TT],rC[1]); \
  TYPE_MAC(rA[0+TT],rB[2+TT],rC[2]); \
  TYPE_MAC(rA[0+TT],rB[3+TT],rC[3]); \
  TYPE_MAC(rA[1+TT],rB[0+TT],rC[4]); \
  TYPE_MAC(rA[1+TT],rB[1+TT],rC[5]); \
  TYPE_MAC(rA[1+TT],rB[2+TT],rC[6]); \
  TYPE_MAC(rA[1+TT],rB[3+TT],rC[7]); \
  TYPE_MAC(rA[2+TT],rB[0+TT],rC[8]); \
  TYPE_MAC(rA[2+TT],rB[1+TT],rC[9]); \
  TYPE_MAC(rA[2+TT],rB[2+TT],rC[10]); \
  TYPE_MAC(rA[2+TT],rB[3+TT],rC[11]); \
  TYPE_MAC(rA[3+TT],rB[0+TT],rC[12]); \
  TYPE_MAC(rA[3+TT],rB[1+TT],rC[13]); \
  TYPE_MAC(rA[3+TT],rB[2+TT],rC[14]); \
  TYPE_MAC(rA[3+TT],rB[3+TT],rC[15]); \

#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) DST = 0 != (BETA) ? (ALPHA)*(REG) + (BETA)*(SRC) : (ALPHA)*(REG)

struct my_timer
{
    struct timeval start_time, end_time;
    double time_use; // us
    
    void start(){
		gettimeofday(&start_time, NULL);
    }

    void stop(){
		gettimeofday(&end_time, NULL);
		time_use = (end_time.tv_sec-start_time.tv_sec)*1.0e6 + end_time.tv_usec-start_time.tv_usec;
    }	
};

#define HIP_CHECK(stat)                                                    \
{                                                                          \
	if(stat != hipSuccess)                                                 \
	{                                                                      \
		std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
		exit(-1);                                                          \
	}                                                                      \
}	

__global__ void global_depth8_lds_2_bank(double *src_a,double *src_b,double *dst_c, double alpha,double beta,int size_m,int size_n,int size_k)
{
    __shared__ double localMemory[LDS_NUM_ELEMENTS];
	
	unsigned int serial = threadIdx.x;//线程号
	unsigned int grj = (serial >> 5);
	unsigned int gri = (serial & 31);
	
	unsigned int goa = grj * size_m + gri * 2;
	unsigned int gob = grj * size_n + gri * 2;
	
	unsigned int lwa = serial * 2;
	unsigned int lwb = serial * 2 + LDS_OFFSET_B;
	
	double *local_write_A =  localMemory + lwa;
	double *local_write_B =  localMemory + lwb;
	
	unsigned int lrj = (serial >> 4);
	unsigned int lri = (serial & 15);
	
	unsigned int lra = lri * 2;
	unsigned int lrb = lrj * 2 + LDS_OFFSET_B;
	
	double *local_read_A = localMemory + lra;
	double *local_read_B = localMemory + lrb;
	
	unsigned int goc = (lri * size_n + lrj ) * 2;
	
	double *global_address_C = dst_c + goc;
	double *global_address_A = src_a + goa;
	double *global_address_B = src_b + gob;
	
	int i,j,k;
	
	double rA[8],rB[8],rC[16];
	double global_a0,global_a1,global_b0,global_b1;
	
	rC[0] = SCALAR_ZERO;
        rC[1] = SCALAR_ZERO;
    	rC[2] = SCALAR_ZERO;
    	rC[3] = SCALAR_ZERO;
    	rC[4] = SCALAR_ZERO;
    	rC[5] = SCALAR_ZERO;
    	rC[6] = SCALAR_ZERO;
    	rC[7] = SCALAR_ZERO;
    	rC[8] = SCALAR_ZERO;
    	rC[9] = SCALAR_ZERO;
    	rC[10] = SCALAR_ZERO;
    	rC[11] = SCALAR_ZERO;
    	rC[12] = SCALAR_ZERO;
    	rC[13] = SCALAR_ZERO;
    	rC[14] = SCALAR_ZERO;
    	rC[15] = SCALAR_ZERO;
		
		//gloabl -> lds
		global_a0 = *(global_address_A + 0); //global read A
		global_a1 = *(global_address_A + 1);
		global_b0 = *(global_address_B + 0); //global read BETA
		global_b1 = *(global_address_B + 1);
		
		global_address_A += DEPTH * size_m;
		global_address_B += DEPTH * size_n;
		
		//write lds
		*(local_write_A + 0) = global_a0;
		*(local_write_A + 1) = global_a1;
		*(local_write_B + 0) = global_b0;
		*(local_write_B + 1) = global_b1;
		
		lwa = (lwa + LDS_OFFSET_BLK) % LDS_NUM_ELEMENTS;
		lwb = (lwb + LDS_OFFSET_BLK) % LDS_NUM_ELEMENTS;
		
		local_write_A =  localMemory + lwa;
		local_write_B =  localMemory + lwb;
		
		__syncthreads();
		
		//preload
		rA[0] = *(local_read_A + 0);
		rA[1] = *(local_read_A + 1);
		rA[2] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0] = *(local_read_B + 0);
		rB[1] = *(local_read_B + 1);
		rB[2] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3] = *(local_read_B + 1 + SIZE_HALF_B);
	
		local_read_A += MT0I;
		local_read_B += MT1J;	
	
	for(i=0;i<size_k;i+=8)
	{	
		//gloabl -> lds
		global_a0 = *(global_address_A + 0); //global read A
		global_a1 = *(global_address_A + 1);
		global_b0 = *(global_address_B + 0); //global read BETA
		global_b1 = *(global_address_B + 1);
		
		global_address_A += DEPTH * size_m;
		global_address_B += DEPTH * size_n;
		
		//iter 0
		rA[0+TT] = *(local_read_A + 0);
		rA[1+TT] = *(local_read_A + 1);
		rA[2+TT] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3+TT] = *(local_read_A + 1 + SIZE_HALF_A);
		
		rB[0+TT] = *(local_read_B + 0);
		rB[1+TT] = *(local_read_B + 1);
		rB[2+TT] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3+TT] = *(local_read_B + 1 + SIZE_HALF_B);	
		
		local_read_A += MT0I;
		local_read_B += MT1J;
		MAC_4x4
		
		//iter 1
		rA[0] = *(local_read_A + 0);
		rA[1] = *(local_read_A + 1);
		rA[2] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0] = *(local_read_B + 0);
		rB[1] = *(local_read_B + 1);
		rB[2] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3] = *(local_read_B + 1 + SIZE_HALF_B);
	
		local_read_A += MT0I;
		local_read_B += MT1J;	
		MAC_4x4_BLK	
		
		//iter 2
		rA[0+TT] = *(local_read_A + 0);
		rA[1+TT] = *(local_read_A + 1);
		rA[2+TT] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3+TT] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0+TT] = *(local_read_B + 0);
		rB[1+TT] = *(local_read_B + 1);
		rB[2+TT] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3+TT] = *(local_read_B + 1 + SIZE_HALF_B);	
		
		local_read_A += MT0I;
		local_read_B += MT1J;	
		MAC_4x4	
		
		//iter 3
		rA[0] = *(local_read_A + 0);
		rA[1] = *(local_read_A + 1);
		rA[2] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0] = *(local_read_B + 0);
		rB[1] = *(local_read_B + 1);
		rB[2] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3] = *(local_read_B + 1 + SIZE_HALF_B);
		
		local_read_A += MT0I;
		local_read_B += MT1J;		
		MAC_4x4_BLK	
		
		//iter 4
		rA[0+TT] = *(local_read_A + 0);
		rA[1+TT] = *(local_read_A + 1);
		rA[2+TT] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3+TT] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0+TT] = *(local_read_B + 0);
		rB[1+TT] = *(local_read_B + 1);
		rB[2+TT] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3+TT] = *(local_read_B + 1 + SIZE_HALF_B);	
		
		local_read_A += MT0I;
		local_read_B += MT1J;	
		MAC_4x4	
		
		//iter 5
		rA[0] = *(local_read_A + 0);
		rA[1] = *(local_read_A + 1);
		rA[2] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0] = *(local_read_B + 0);
		rB[1] = *(local_read_B + 1);
		rB[2] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3] = *(local_read_B + 1 + SIZE_HALF_B);
		
		local_read_A += MT0I;
		local_read_B += MT1J;
		MAC_4x4_BLK	
		
		//iter 6
		rA[0+TT] = *(local_read_A + 0);
		rA[1+TT] = *(local_read_A + 1);
		rA[2+TT] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3+TT] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0+TT] = *(local_read_B + 0);
		rB[1+TT] = *(local_read_B + 1);
		rB[2+TT] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3+TT] = *(local_read_B + 1 + SIZE_HALF_B);		
		
		//write lds
		*(local_write_A + 0) = global_a0;
		*(local_write_A + 1) = global_a1;
		*(local_write_B + 0) = global_b0;
		*(local_write_B + 1) = global_b1;
		
		lwa = ( lwa + LDS_OFFSET_BLK ) % LDS_NUM_ELEMENTS;
		lwb = ( lwb + LDS_OFFSET_BLK ) % LDS_NUM_ELEMENTS;
		
		local_write_A =  localMemory + lwa;
		local_write_B =  localMemory + lwb ;
		
		lra = (lra + LDS_OFFSET_BLK) % LDS_NUM_ELEMENTS;
		lrb = (lrb + LDS_OFFSET_BLK) % LDS_NUM_ELEMENTS;
		
		local_read_A = localMemory + lra;
		local_read_B = localMemory + lrb;
		
		MAC_4x4
		__syncthreads();
		
		//preload
		rA[0] = *(local_read_A + 0);
		rA[1] = *(local_read_A + 1);
		rA[2] = *(local_read_A + 0 + SIZE_HALF_A);
		rA[3] = *(local_read_A + 1 + SIZE_HALF_A);
			
		rB[0] = *(local_read_B + 0);
		rB[1] = *(local_read_B + 1);
		rB[2] = *(local_read_B + 0 + SIZE_HALF_B);
		rB[3] = *(local_read_B + 1 + SIZE_HALF_B);
	
		local_read_A += MT0I;
		local_read_B += MT1J;	
		
		MAC_4x4_BLK

	}
	
	
	TYPE_MAC_WRITE(*(global_address_C+0), *(global_address_C+0), alpha, rC[0], beta);
	TYPE_MAC_WRITE(*(global_address_C+1), *(global_address_C+1), alpha, rC[1], beta);
	TYPE_MAC_WRITE(*(global_address_C+0+SIZE_HALF_C), *(global_address_C+0+SIZE_HALF_C), alpha, rC[2], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+SIZE_HALF_C), *(global_address_C+1+SIZE_HALF_C), alpha, rC[3], beta);
	
	TYPE_MAC_WRITE(*(global_address_C+0+size_n), *(global_address_C+0+size_n), alpha, rC[4], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+size_n), *(global_address_C+1+size_n), alpha, rC[5], beta);
	TYPE_MAC_WRITE(*(global_address_C+0+SIZE_HALF_C+size_n), *(global_address_C+0+SIZE_HALF_C+size_n), alpha, rC[6], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+SIZE_HALF_C+size_n), *(global_address_C+1+SIZE_HALF_C+size_n), alpha, rC[7], beta);
	
	TYPE_MAC_WRITE(*(global_address_C+0+size_n*SIZE_HALF_C), *(global_address_C+0+size_n*SIZE_HALF_C), alpha, rC[8], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+size_n*SIZE_HALF_C), *(global_address_C+1+size_n*SIZE_HALF_C), alpha, rC[9], beta);
	TYPE_MAC_WRITE(*(global_address_C+0+SIZE_HALF_C+size_n*SIZE_HALF_C), *(global_address_C+0+SIZE_HALF_C+size_n*SIZE_HALF_C), alpha, rC[10], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+SIZE_HALF_C+size_n*SIZE_HALF_C), *(global_address_C+1+SIZE_HALF_C+size_n*SIZE_HALF_C), alpha, rC[11], beta);
	
	TYPE_MAC_WRITE(*(global_address_C+0+size_n*(SIZE_HALF_C+1)), *(global_address_C+0+size_n*(SIZE_HALF_C+1)), alpha, rC[12], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+size_n*(SIZE_HALF_C+1)), *(global_address_C+1+size_n*(SIZE_HALF_C+1)), alpha, rC[13], beta);
	TYPE_MAC_WRITE(*(global_address_C+0+SIZE_HALF_C+size_n*(SIZE_HALF_C+1)), *(global_address_C+0+SIZE_HALF_C+size_n*(SIZE_HALF_C+1)), alpha, rC[14], beta);
	TYPE_MAC_WRITE(*(global_address_C+1+SIZE_HALF_C+size_n*(SIZE_HALF_C+1)), *(global_address_C+1+SIZE_HALF_C+size_n*(SIZE_HALF_C+1)), alpha, rC[15], beta);
	
}

void mul_cpu(double *src_a,double *src_b,double *dst_c, double alpha,double beta,int size_m,int size_n,int size_k)
{
	int i,j,k;
	for(i=0;i<size_m;i++)
	{
		for(j=0;j<size_n;j++)
		{
			double sum=0;
			for(k=0;k<size_k;k++)
			{
				sum += src_a[k*size_m+i] * src_b[k*size_n+j];
			}
			dst_c[i*size_n+j] = alpha*sum+beta*dst_c[i*size_n+j];
		}
	}
}

int main()
{
	double *src_a;
	double *src_b;
	double *out_cpu;
	double *out_cpu2;
	double *a_device, *b_device, *c_device;
	double *out_gpu;
	double alpha = 2.0;
	double beta  = 3.0;
	int size_m   = 64;
	int size_n   = 64;
	int size_k   = 128;

	int error = 0;
	
	src_a   = (double *)malloc(size_m*size_k*sizeof(double));
	assert(src_a != NULL);
	src_b   = (double *)malloc(size_k*size_n*sizeof(double));
	assert(src_b != NULL);
	out_cpu = (double *)malloc(size_m*size_n*sizeof(double));
	assert(out_cpu != NULL);
	out_gpu = (double *)malloc(size_m*size_n*sizeof(double));
	assert(out_gpu != NULL);
	out_cpu2 = (double *)malloc(size_m*size_n*sizeof(double));
	assert(out_cpu2 != NULL);

	HIP_CHECK( hipMalloc((void**)&a_device, size_m*size_k*sizeof(double)) );
	HIP_CHECK( hipMalloc((void**)&b_device, size_k*size_n*sizeof(double)) );
	HIP_CHECK( hipMalloc((void**)&c_device, size_m*size_n*sizeof(double)) );

	for(int i=0; i<size_m*size_k; i++)
	{
		src_a[i] = rand()%128;
	}

	for(int i=0; i<size_k*size_n; i++)
	{
		src_b[i] = rand()%128;
	}
	
	for(int i=0; i<size_m*size_n; i++)
	{
		out_gpu[i] = out_cpu[i] = out_cpu2[i] = rand()%128;
	}

	HIP_CHECK( hipMemcpy(a_device, src_a,   size_m*size_k*sizeof(double), hipMemcpyHostToDevice) );
	HIP_CHECK( hipMemcpy(b_device, src_b,   size_n*size_k*sizeof(double), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(c_device, out_cpu, size_m*size_n*sizeof(double), hipMemcpyHostToDevice) );

#ifdef WARMUP
	for(int iter=0; iter<5; ++iter)
	{
		hipLaunchKernelGGL(global_depth8_lds_2_bank, dim3(1,1,1), dim3(256,1,1), 0, 0, a_device, b_device, c_device, alpha, beta, size_m, size_n, size_k);
		HIP_CHECK( hipMemcpy(c_device, out_cpu, size_m*size_n*sizeof(double), hipMemcpyHostToDevice) );
	}	
#endif

	double sum_costs1 = 0.0;
	my_timer timer1;
	for(int iter=0; iter<10; ++iter)
	{
		HIP_CHECK( hipMemcpy(c_device, out_cpu, size_m*size_n*sizeof(double), hipMemcpyHostToDevice) );
		hipDeviceSynchronize();
		timer1.start();
		hipLaunchKernelGGL(global_depth8_lds_2_bank, dim3(1,1,1), dim3(256,1,1), 0, 0, a_device, b_device, c_device, alpha, beta, size_m, size_n, size_k);
		hipDeviceSynchronize();
		timer1.stop();
		sum_costs1 += timer1.time_use;
	}
	HIP_CHECK( hipMemcpy(out_gpu, c_device, size_m*size_n*sizeof(double), hipMemcpyDeviceToHost) );

	double sum_costs2 = 0.0;
	my_timer timer2;
	for(int iter=0; iter<10; ++iter)
	{
		memcpy(out_cpu, out_cpu2, size_m*size_n*sizeof(double));
		timer2.start();
		mul_cpu(src_a, src_b, out_cpu, alpha, beta, size_m, size_n, size_k);
		timer2.stop();
		sum_costs2 += timer2.time_use;
	}

	for(int i=0; i<size_m*size_n; i++)
	{
		if(fabs(out_gpu[i]-out_cpu[i]) > 1.0e-6)
		{
			error++;
			printf("%d,%lf,%lf\n",i,out_gpu[i],out_cpu[i]);
		}
	}
	
	if(error == 0)
	{
		printf("************************result is ok\n");
	}
		
	double run_time_cpu = (sum_costs2/10)/1000.0;
	printf("cpu time is %f (ms), GFlops is %f GFlops\n\n",run_time_cpu, (double)(2*size_m*size_n*size_k)/(run_time_cpu*1000000));

	double run_time_dcu = (sum_costs1/10)/1000.0;
	printf("dcu time is %f (ms), GFlops is %f GFlops\n\n",run_time_dcu, (double)(2*size_m*size_n*size_k)/(run_time_dcu*1000000));
	
	printf("finish\n");

	free(src_a);
	free(src_b);
	free(out_cpu);
	free(out_cpu2);
	free(out_gpu);

	hipFree(a_device);
	hipFree(b_device);
	hipFree(c_device);

	return 0;
}
