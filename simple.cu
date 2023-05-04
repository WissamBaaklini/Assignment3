#include <stdio.h>
#include <cuda_profiler_api.h>

#define M 1024
#define N 1024

__global__ void matrixMul(int *A, int *B, int *C)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

int main()
{
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int size = M * N * sizeof(int);

    // Allocate memory on the host
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i + j;
            B[i * N + j] = i * j;
        }
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // The number of threads used by the CUDA C code for matrix multiplication depends on
    // the block and grid dimensions that are specified when launching the kernel function.
    // In this code, the block dimensions are set to 16x16 threads, so each block contains 256 threads.
    // The grid dimensions are calculated based on the size of the input matrices and the block dimensions.
    // Specifically, the number of blocks in the x and y directions are calculated as follows
    // Define grid and block dimensions
    dim3 dimGrid((N - 1) / 16 + 1, (M - 1) / 16 + 1, 1);
    dim3 dimBlock(16, 16, 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Launch kernel function
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy output matrix from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print execution time
    printf("Execution time: %f seconds\n", elapsedTime / 1000.0);

    // Free memory on the host and device
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

