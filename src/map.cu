#include <stdexcept>
#include "base64/base64.h"
#include "util/cuda_assert.cuh"
#include "util/reader.h"
#include "map.cuh"
#include "chunkupdates.cuh"
#include "render.cuh"

__global__ void getArrow(cudarrows::Chunk *chunks, int16_t chunkX, int16_t chunkY, int8_t x, int8_t y, cudarrows::ArrowInfo *arrowInfo) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    uint8_t idx = y * CHUNK_SIZE + x;
    cudarrows::Arrow &arrow = chunk.arrows[idx];
    arrowInfo->type = arrow.type;
    arrowInfo->rotation = arrow.rotation;
    arrowInfo->flipped = arrow.flipped;
}

__global__ void sendInput(cudarrows::Chunk *chunks, int32_t x, int32_t y, cudarrows::ArrowInput input) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    if (x / CHUNK_SIZE != chunk.x || y / CHUNK_SIZE != chunk.y) return;
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    cudarrows::Arrow &arrow = chunk.arrows[idx];
    arrow.input = input;
}

cudarrows::Map::Map(const std::string &save) {
    std::string buf = base64_decode(save);
    if (buf.size() < 4) return;
    util::Reader reader(buf);
    if (reader.read16() != 0)
        throw std::invalid_argument("Unsupported save version");
    chunkCount = reader.read16();
    cudarrows::Chunk *h_chunks = new cudarrows::Chunk[chunkCount];
    for (uint16_t i = 0; i < chunkCount; ++i) {
        cudarrows::Chunk &chunk = h_chunks[i];
        chunk.x = reader.read16();
        chunk.y = reader.read16();
        uint8_t arrowTypeCount = reader.read8();
        for (uint8_t j = 0; j <= arrowTypeCount; ++j) {
            uint8_t type = reader.read8();
            uint8_t arrowCount = reader.read8();
            for (uint8_t k = 0; k <= arrowCount; ++k) {
                uint8_t position = reader.read8();
                uint8_t arrowX = position & 0xF;
                uint8_t arrowY = position >> 4;
                uint8_t rotation = reader.read8();
                cudarrows::Arrow &arrow = chunk.arrows[arrowY * CHUNK_SIZE + arrowX];
                arrow.type = (cudarrows::ArrowType)type;
                arrow.rotation = (cudarrows::ArrowRotation)(rotation & 0x3);
                arrow.flipped = rotation & 0x4;
                if (k == 255) break;
            }
            if (j == 255) break;
        }
    }
    cuda_assert(cudaMalloc(&chunks, sizeof(cudarrows::Chunk) * chunkCount));
    cuda_assert(cudaMemcpy(chunks, h_chunks, sizeof(cudarrows::Chunk) * chunkCount, cudaMemcpyHostToDevice));
    delete h_chunks;
}

cudarrows::Map::~Map() {
    cuda_assert(cudaFree(chunks));
}

cudarrows::ArrowInfo cudarrows::Map::getArrow(int32_t x, int32_t y) {
    int16_t chunkX = x / CHUNK_SIZE;
    int16_t chunkY = y / CHUNK_SIZE;
    x -= chunkX * CHUNK_SIZE;
    y -= chunkY * CHUNK_SIZE;
    cudarrows::ArrowInfo h_arrow, *d_arrow;
    cudaMalloc(&d_arrow, sizeof(cudarrows::ArrowInfo));
    size_t numThreads = chunkCount > 1024 ? 1024 : chunkCount;
    size_t numBlocks = (chunkCount + numThreads - 1) / chunkCount;
    ::getArrow<<<numBlocks, numThreads>>>(chunks, chunkX, chunkY, x, y, d_arrow);
    cudaMemcpy(&h_arrow, d_arrow, sizeof(cudarrows::ArrowInfo), cudaMemcpyDeviceToHost);
    cudaFree(d_arrow);
    return h_arrow;
}

void cudarrows::Map::sendInput(int32_t x, int32_t y, cudarrows::ArrowInput input) {
    ::sendInput<<<chunkCount, dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(chunks, x, y, input);
}

void cudarrows::Map::reset(uint64_t seed) {
    ::reset<<<dim3(chunkCount, 2), dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(chunks, seed);
}

void cudarrows::Map::update() {
    std::swap(step, nextStep);
    ::update<<<chunkCount, dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(chunks, step, nextStep);
}

void cudarrows::Map::render(cudaSurfaceObject_t surface, int32_t minX, int32_t minY, int32_t maxX, int32_t maxY) {
    ::render<<<chunkCount, dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(surface, chunks, step, minX, minY, maxX, maxY);
}