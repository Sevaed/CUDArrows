#include "render.h"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, int32_t minX, int32_t minY, int32_t maxX, int32_t maxY) {
    cudarrows::Chunk chunk = chunks[blockIdx.x];
    int16_t chunkX = chunk.x;
    int16_t chunkY = chunk.y;
    int32_t x = chunkX * CHUNK_SIZE + threadIdx.x;
    int32_t y = chunkY * CHUNK_SIZE + threadIdx.y;
    if (x < minX || y < minY || x > maxX || y > maxY) return;
    x -= minX;
    y -= minY;
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    cudarrows::Arrow arrow = chunk.arrows[idx];
    cudarrows::ArrowState state = chunk.states[step][idx];
    uchar4 data = { arrow.type, arrow.rotation + 0x4 * arrow.flipped, state.signal, 255 };
    surf2Dwrite(data, surf, x * sizeof(data), y);
}