#include <stdio.h>
#include <cuda_profiler_api.h>

#define M 1024
#define N 1024
#define TILE_SIZE 32

// The key changes from simple matrix multiplcation code are:

// Introducing a TILE_SIZE constant to specify the size of the tile.
// Using thread indices and block indices to compute the row and column of the current element being computed.
// Declaring shared memory for each input matrix.
// Using a loop over tiles to load the input matrices into shared memory and perform the computation.
// Using __syncthreads() to synchronize threads within a block during the computation.

__global__ void matrixMul(int *A, int *B, int *C)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ int s_A[TILE_SIZE][TILE_SIZE];
    __shared__ int s_B[TILE_SIZE][TILE_SIZE];

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    int sum = 0;
    for (int t = 0; t < (N - 1) / TILE_SIZE + 1; t++)
    {
        if (row < M && t * TILE_SIZE + tx < N)
        {
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        }
        else
        {
            s_A[ty][tx] = 0;
        }
        if (col < N && t * TILE_SIZE + ty < M)
        {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        }
        else
        {
            s_B[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
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

    // Define grid and block dimensions
    dim3 dimGrid((N - 1) / TILE_SIZE + 1, (M - 1) / TILE_SIZE + 1, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

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