#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CHUNK_SIZE 16

struct Arrow {
    uint8_t type;
};

struct Chunk {
    uint16_t x, y;
    Arrow arrows[CHUNK_SIZE * CHUNK_SIZE][2];
};

__global__ void update(Chunk *chunks, unsigned long long steps) {
    for (unsigned long long i = 0; i < steps; i++) {
        // ...
        __syncthreads();
    }
}

/*
thrust::host_vector<Chunk> h_vec;
h_vec.push_back(Chunk { 0, 0 });
h_vec.push_back(Chunk { 10, 0 });
h_vec.push_back(Chunk { 10, 12 });
thrust::device_vector<Chunk> d_vec = h_vec;
Chunk *chunks = thrust::raw_pointer_cast(d_vec.data());
clock_t start = clock();
unsigned long long i = 0;
while ((clock() - start) < 5000) {
    update<<<d_vec.size(), dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(chunks);
    i += 100000;
}
cudaDeviceSynchronize();
std::cout << (i / float(clock() - start)) << " iterations per second" << std::endl;
*/