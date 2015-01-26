#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include <stdio.h>
#include <iostream>

#define CacheCount 3

__global__ void BenchMarkDRAMKernel(float4* In) {
    int ThreadID = blockDim.x *blockIdx.x + threadIdx.x ;

    float4 Temp = make_float4(1);

    Temp += In[ThreadID];

    if (length(Temp) == -12354) {
        In[0] = Temp;
    }
}

__global__ void BenchMarkCacheKernel(float4* In, int Zero) {
    int ThreadID = blockDim.x *blockIdx.x + threadIdx.x;

    float4 Temp = make_float4(1);

    #pragma unroll
    for (int i = 0; i < CacheCount; i++) {
        Temp += In[ThreadID + i*Zero];
    }

    if (length(Temp) == -12354) {
        In[0] = Temp;
    }
}

int main() {
    static const int PointerCount = 5000;

    int Float4Count = 4 * 1024 * 1024;
    int ChunkSize = Float4Count*sizeof(float4);
    float4* Pointers[PointerCount];
    int UsedPointers = 0;
    printf("Nai's Benchmark \n");
    printf("Allocating Memory . . . Chunk Size = %i Byte \n", ChunkSize);
    system("pause");

    while (true) {
        int Error = cudaMalloc(&Pointers[UsedPointers], ChunkSize);

        if (Error == cudaErrorMemoryAllocation)
        break;

        printf("Allocated at %llx\n", (long long unsigned)Pointers[UsedPointers]);

        cudaMemset(Pointers[UsedPointers], 0, ChunkSize);
        UsedPointers++;
    }

    printf("Allocated %i Chunks \n", UsedPointers);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int BlockSize = 128;
    int BlockCount = Float4Count / BlockSize;

    int BenchmarkCount = 10;

    printf("Benchmarking DRAM \n");

    for (int i = 0; i < UsedPointers; i++) {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++)
            BenchMarkDRAMKernel <<<BlockCount, BlockSize >>>(Pointers[i]);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float Bandwidth = ((float)(BenchmarkCount * ChunkSize)) / milliseconds/ 1000.f/1000.f;
        printf("DRAM-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    printf("Copying between low and low chunks\n");

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++) {
            cudaMemcpy(Pointers[i], Pointers[i + 10], ChunkSize, cudaMemcpyDeviceToDevice);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        float Bandwidth = ((float)(BenchmarkCount * ChunkSize)) / ms/ 1000.f/1000.f;
        printf("DRAM-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    printf("Copying between high and low chunks\n");

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++) {
            cudaMemcpy(Pointers[i], Pointers[UsedPointers - i - 1], ChunkSize, cudaMemcpyDeviceToDevice);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        float Bandwidth = ((float)(BenchmarkCount * ChunkSize)) / ms/ 1000.f/1000.f;
        printf("DRAM-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    printf("Copying between high and high chunks\n");

    for (int i = 0; i < 4; i++) {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++) {
            cudaMemcpy(Pointers[UsedPointers - i - 5], Pointers[UsedPointers - i - 1], ChunkSize, cudaMemcpyDeviceToDevice);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        float Bandwidth = ((float)(BenchmarkCount * ChunkSize)) / ms/ 1000.f/1000.f;
        printf("DRAM-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    printf("Freeing first chunks\n");

    // Free the first half of memory
    for (int i = 0; i < UsedPointers / 2; i++) {
        int error = cudaFree(Pointers[UsedPointers]);
        if (error != cudaSuccess) {
            printf("Error freeing memory\n");
            exit(-1);
        }
    }

    printf("Re-benchmarking upper half of memory\n");

    for (int i = UsedPointers / 2; i < UsedPointers; i++) {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++)
        BenchMarkDRAMKernel <<<BlockCount, BlockSize >>>(Pointers[i]);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float Bandwidth = ((float)(BenchmarkCount * ChunkSize)) / milliseconds/ 1000.f/1000.f;
        printf("DRAM-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    printf("Exiting early.");
    exit(0);

    printf("Benchmarking L2-Cache \n");

    for (int i = 0; i < UsedPointers; i++)
    {
        cudaEventRecord(start);
        for (int j = 0; j < BenchmarkCount; j++) {
            BenchMarkCacheKernel << <BlockCount, BlockSize >> >(Pointers[i], 0);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float Bandwidth = (((float)CacheCount* (float)BenchmarkCount * (float)ChunkSize)) / milliseconds / 1000.f / 1000.f;
        printf("L2-Cache-Bandwidth of %i. Chunk: %f GByte/s \n", i, Bandwidth);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
