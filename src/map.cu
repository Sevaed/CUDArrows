#include <stdexcept>
#include "base64/base64.h"
#include "util/cuda_assert.cuh"
#include "util/reader.h"
#include "map.cuh"
#include "chunkupdates.cuh"
#include "render.cuh"

__global__ void getArrow(cudarrows::Chunk *chunks, size_t count, int16_t chunkX, int16_t chunkY, int8_t x, int8_t y, cudarrows::ArrowInfo *arrowInfo) {
    size_t idx = blockIdx.x * 1024 + threadIdx.x;
    if (idx >= count) return;
    cudarrows::Chunk &chunk = chunks[idx];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    cudarrows::Arrow &arrow = chunk.arrows[y * CHUNK_SIZE + x];
    arrowInfo->type = arrow.type;
    arrowInfo->rotation = arrow.rotation;
    arrowInfo->flipped = arrow.flipped;
}

__global__ void sendInput(cudarrows::Chunk *chunks, size_t count, int16_t chunkX, int16_t chunkY, int8_t x, int8_t y, cudarrows::ArrowInput input) {
    size_t idx = blockIdx.x * 1024 + threadIdx.x;
    if (idx >= count) return;
    cudarrows::Chunk &chunk = chunks[idx];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    cudarrows::Arrow &arrow = chunk.arrows[y * CHUNK_SIZE + x];
    arrow.input = input;
}

__global__ void linkChunks(cudarrows::Chunk *chunks, size_t count) {
    size_t idx1 = blockIdx.x * 32 + threadIdx.x;
    size_t idx2 = blockIdx.y * 32 + threadIdx.y;
    if (idx1 >= count || idx2 >= count || idx1 == idx2) return;
    cudarrows::Chunk &chunkA = chunks[idx1];
    cudarrows::Chunk &chunkB = chunks[idx2];
    int16_t adjacentChunksCoords[8][2] = {
        { chunkA.x,     chunkA.y - 1 },
        { chunkA.x + 1, chunkA.y - 1 },
        { chunkA.x + 1, chunkA.y     },
        { chunkA.x + 1, chunkA.y + 1 },
        { chunkA.x,     chunkA.y + 1 },
        { chunkA.x - 1, chunkA.y + 1 },
        { chunkA.x - 1, chunkA.y     },
        { chunkA.x - 1, chunkA.y - 1 },
    };
    for (uint8_t i = 0; i < sizeof(adjacentChunksCoords) / sizeof(adjacentChunksCoords[0]); ++i)
        if (adjacentChunksCoords[i][0] == chunkB.x && adjacentChunksCoords[i][1] == chunkB.y) {
            chunkA.adjacentChunks[i] = &chunkB;
            return;
        }
}

cudarrows::Map::Map(const std::string &save) {
    std::string buf = base64_decode(save);
    if (buf.size() < 4) return;
    util::Reader reader(buf);
    if (reader.readU16() != 0)
        throw std::invalid_argument("Unsupported save version");
    chunkCount = reader.readU16();
    cudarrows::Chunk *h_chunks = new cudarrows::Chunk[chunkCount];
    for (uint16_t i = 0; i < chunkCount; ++i) {
        cudarrows::Chunk &chunk = h_chunks[i];
        chunk.x = reader.readI16();
        chunk.y = reader.readI16();
        uint8_t arrowTypeCount = reader.readU8();
        for (uint8_t j = 0; j <= arrowTypeCount; ++j) {
            uint8_t type = reader.readU8();
            uint8_t arrowCount = reader.readU8();
            for (uint8_t k = 0; k <= arrowCount; ++k) {
                uint8_t position = reader.readU8();
                uint8_t arrowX = position & 0xF;
                uint8_t arrowY = position >> 4;
                uint8_t rotation = reader.readU8();
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
    unsigned int numBlocks = (chunkCount + 31) / 32;
    ::linkChunks<<<dim3(numBlocks, numBlocks), dim3(32, 32)>>>(chunks, chunkCount);
    cuda_assert(cudaDeviceSynchronize());
}

cudarrows::Map::~Map() {
    cuda_assert(cudaFree(chunks));
}

cudarrows::ArrowInfo cudarrows::Map::getArrow(int32_t x, int32_t y) {
    int16_t chunkX = arrowToChunk(x);
    int16_t chunkY = arrowToChunk(y);
    x -= chunkX * CHUNK_SIZE;
    y -= chunkY * CHUNK_SIZE;
    cudarrows::ArrowInfo h_arrow, *d_arrow;
    cuda_assert(cudaMalloc(&d_arrow, sizeof(cudarrows::ArrowInfo)));
    unsigned int numBlocks = (chunkCount + 1023) / 1024;
    ::getArrow<<<numBlocks, 1024>>>(chunks, chunkCount, chunkX, chunkY, x, y, d_arrow);
    cuda_assert(cudaDeviceSynchronize());
    cuda_assert(cudaMemcpy(&h_arrow, d_arrow, sizeof(cudarrows::ArrowInfo), cudaMemcpyDeviceToHost));
    cuda_assert(cudaFree(d_arrow));
    return h_arrow;
}

void cudarrows::Map::sendInput(int32_t x, int32_t y, cudarrows::ArrowInput input) {
    int16_t chunkX = arrowToChunk(x);
    int16_t chunkY = arrowToChunk(y);
    x -= chunkX * CHUNK_SIZE;
    y -= chunkY * CHUNK_SIZE;
    unsigned int numBlocks = (chunkCount + 1023) / 1024;
    ::sendInput<<<numBlocks, 1024>>>(chunks, chunkCount, chunkX, chunkY, x, y, input);
    cuda_assert(cudaDeviceSynchronize());
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

int16_t cudarrows::Map::arrowToChunk(int32_t x) {
    int32_t neg = x < 0 ? 1 : 0;
    return (x + neg) / CHUNK_SIZE - neg;
}