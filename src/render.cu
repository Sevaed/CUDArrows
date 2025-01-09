#include "render.h"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, int32_t minX, int32_t minY, int32_t maxX, int32_t maxY) {
    const cudarrows::Chunk &chunk = chunks[blockIdx.x];
    int32_t x = chunk.x * CHUNK_SIZE + threadIdx.x;
    int32_t y = chunk.y * CHUNK_SIZE + threadIdx.y;
    if (x < minX || y < minY || x > maxX || y > maxY) return;
    x -= minX;
    y -= minY;
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    const cudarrows::Arrow &arrow = chunk.arrows[idx];
    uchar4 data = { arrow.type, arrow.rotation + (uint8_t)0x4 * arrow.flipped, arrow.state[step].signal, 255 };
    surf2Dwrite(data, surf, x * sizeof(data), y);
}