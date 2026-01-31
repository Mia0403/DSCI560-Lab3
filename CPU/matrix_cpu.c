#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void matrixMultiplyCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed for N=%d (need %.2f MB per matrix)\n",
                N, (double)size / (1024.0 * 1024.0));
        free(A); free(B); free(C);
        return 1;
    }

    // init with pseudo-random floats in [0, 0.99]
    srand(0); // fixed seed for reproducibility
    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 100) / 100.0f;
        B[i] = (rand() % 100) / 100.0f;
    }

    clock_t start = clock();
    matrixMultiplyCPU(A, B, C, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU execution time (N=%d): %.6f seconds\n", N, elapsed);

    free(A); free(B); free(C);
    return 0;
}