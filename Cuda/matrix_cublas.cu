#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);

    srand(0);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (rand() % 100) / 100.0f;
        hB[i] = (rand() % 100) / 100.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Row-major C = A*B  <=>  Column-major C^T = B^T * A^T
    // Using column-major convention with no transpose, swap A and B pointers:
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                dB, N,
                dA, N,
                &beta,
                dC, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    printf("cuBLAS SGEMM time (N=%d): %.3f ms (%.6f seconds)\n", N, ms, ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
