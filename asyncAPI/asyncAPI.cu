/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */

// includes, system
#include <stdio.h>
#include <chrono>

// includes CUDA Runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

template<typename T>
void __do_check(T result, const char* funcname, const char* filename, const int linenum) {
    if (result) {
        // cudaGetErrorName:
        // Turn an error code into an error string for rendering on the stdout
        fprintf(stderr, "cuda error %d (%s) at %s: %d\n", static_cast<int>(result), cudaGetErrorName(result), filename, linenum);
        exit(-1);
    }
}

// Some macros for checking cuda API error.
// Most cuda APIs return 
#define CHECK_ERROR(val) __do_check((val), #val, __FILE__, __LINE__)


__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
        if (data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int main(int argc, char *argv[])
{
    int            devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);    
    
    printf("counting available devices on this machine: \n");
    int device_count = 0;
    CHECK_ERROR(cudaGetDeviceCount(&device_count));
    printf("%d cuda device detected on this machine\n", device_count);

    // This will pick the best possible CUDA capable device
    if (device_count > 0) {
        printf("using device 0 by default\n");
        devID = 0;
    }
    else {
        printf("no available cuda device on this machine\n");
        exit(-1);
    }

    // get device name
    CHECK_ERROR(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    CHECK_ERROR(cudaSetDevice(devID));

    int n      = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value  = 26;

    // alloc host memory using cudaMallocHost
    // in theory it can also be done with stdlib's malloc
    // But it seems that stdlib malloc does not pin the memory pages
    // and does not work well with asynchronous host-device memory copy
    int *a_host = 0;
    CHECK_ERROR(cudaMallocHost((void**)&a_host, nbytes));

    int *a_dev = 0;
    CHECK_ERROR(cudaMalloc((void**)&a_dev, nbytes));

    // After doing the allocation, set the initial values
    memset(a_host, 0, nbytes);
    cudaMemset((void*)a_dev, 0, nbytes);

    CHECK_ERROR(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    dim3 thread(512, 1);
    dim3 block(n/thread.x, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    auto start_time = std::chrono::steady_clock::now();

    // The following 5 steps submit 5 tasks to the GPU, in asynchronous fashion.
    cudaEventRecord(start, 0);
    cudaMemcpyAsync((void*) a_dev, (void*) a_host, nbytes, cudaMemcpyKind::cudaMemcpyHostToDevice, 0);
    increment_kernel<<<block, thread, 0, 0>>>(a_dev, value);
    cudaMemcpyAsync((void*) a_host, (void*) a_dev, nbytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    
    // Here we only record the overhead for submitting to GPU, not the actual task running time.
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    
    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        // We can check from the CPU side for whether the task has finished running.
        counter++;
    }

    // The `stop` event has finally be executed by the GPU. This is
    // the actual end of the work. Measure the actual time that GPU spent
    // to run the submitted task, including bi-directional memory copies between
    // host and device and the kernel execution time.
    CHECK_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));

    printf("execution time on host: %.2f\n", elapsed.count());
    printf("execution time on gpu: %.2f\n", gpu_time);

    cudaFreeHost((void*) a_host);
    cudaFree((void*)a_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
