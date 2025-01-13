#include <stdio.h>

static void cuda_check(const char *file, int line, cudaError_t error) {
    if (error != cudaError::cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s:%d) %d: %s\n", file, line, error, cudaGetErrorString(error));
        abort();
    }
}

#define cuda_assert(error) cuda_check(__FILE__, __LINE__, error)