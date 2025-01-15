#include "render.cuh"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, cudarrows::globalCoord minX, cudarrows::globalCoord minY, cudarrows::globalCoord maxX, cudarrows::globalCoord maxY) {
    const cudarrows::Chunk &chunk = chunks[blockIdx.x];
    cudarrows::globalCoord x = chunk.x * CHUNK_SIZE + threadIdx.x;
    cudarrows::globalCoord y = chunk.y * CHUNK_SIZE + threadIdx.y;
    if (x < minX || y < minY || x > maxX || y > maxY) return;
    x -= minX;
    y -= minY;
    cudarrows::arrowIdx idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    const cudarrows::Arrow &arrow = chunk.arrows[idx];
    uchar4 data = { (uint8_t)arrow.type, (uint8_t)arrow.rotation + (uint8_t)0x4 * arrow.flipped, (uint8_t)arrow.state[step].signal, 255 };
    surf2Dwrite(data, surf, x * sizeof(data), y);
}