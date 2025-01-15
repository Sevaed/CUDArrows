#include <stdexcept>
#include "base64/base64.h"
#include "util/cuda_assert.cuh"
#include "util/reader.h"
#include "map.cuh"
#include "chunkupdates.cuh"
#include "render.cuh"

__global__ void getArrow(cudarrows::Chunk *chunks, size_t count, cudarrows::chunkCoord chunkX, cudarrows::chunkCoord chunkY, cudarrows::localCoord x, cudarrows::localCoord y, cudarrows::ArrowInfo *arrowInfo) {
    size_t idx = blockIdx.x * 1024 + threadIdx.x;
    if (idx >= count) return;
    cudarrows::Chunk &chunk = chunks[idx];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    cudarrows::Arrow &arrow = chunk.arrows[y * CHUNK_SIZE + x];
    arrowInfo->type = arrow.type;
    arrowInfo->rotation = arrow.rotation;
    arrowInfo->flipped = arrow.flipped;
}

__global__ void sendInput(cudarrows::Chunk *chunks, size_t count, cudarrows::chunkCoord chunkX, cudarrows::chunkCoord chunkY, cudarrows::localCoord x, cudarrows::localCoord y, cudarrows::ArrowInput input) {
    size_t idx = blockIdx.x * 1024 + threadIdx.x;
    if (idx >= count) return;
    cudarrows::Chunk &chunk = chunks[idx];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    cudarrows::Arrow &arrow = chunk.arrows[y * CHUNK_SIZE + x];
    arrow.input = input;
}

__global__ void fillChunk(cudarrows::Chunk *chunks, cudarrows::chunkCoord chunkX, cudarrows::chunkCoord chunkY, cudarrows::localCoord offsetX, cudarrows::localCoord offsetY, cudarrows::localCoord width, cudarrows::ArrowInfo *arrows, bool *success) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    if (chunkX != chunk.x || chunkY != chunk.y) return;
    size_t srcIdx = threadIdx.y * width + threadIdx.x;
    size_t dstIdx = (offsetY + threadIdx.y) * CHUNK_SIZE + (offsetX + threadIdx.x);
    cudarrows::ArrowInfo &src = arrows[srcIdx];
    cudarrows::Arrow &dst = chunk.arrows[dstIdx];
    dst.type = src.type;
    dst.rotation = src.rotation;
    dst.flipped = src.flipped;
    *success = true;
}

__global__ void initChunk(cudarrows::Chunk *chunks, size_t idx, cudarrows::chunkCoord chunkX, cudarrows::chunkCoord chunkY, cudarrows::localCoord offsetX, cudarrows::localCoord offsetY, cudarrows::localCoord width, cudarrows::ArrowInfo *arrows) {
    cudarrows::Chunk &chunk = chunks[idx];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        chunk.x = chunkX;
        chunk.y = chunkY;
    }
    size_t srcIdx = threadIdx.y * width + threadIdx.x;
    size_t dstIdx = (offsetY + threadIdx.y) * CHUNK_SIZE + (offsetX + threadIdx.x);
    cudarrows::ArrowInfo &src = arrows[srcIdx];
    cudarrows::Arrow &dst = chunk.arrows[dstIdx];
    dst.type = src.type;
    dst.rotation = src.rotation;
    dst.flipped = src.flipped;
}

__global__ void linkChunks(cudarrows::Chunk *chunks, size_t count) {
    size_t idx1 = blockIdx.x * 32 + threadIdx.x;
    size_t idx2 = blockIdx.y * 32 + threadIdx.y;
    if (idx1 >= count || idx2 >= count/* || idx1 == idx2*/) return;
    cudarrows::Chunk &chunkA = chunks[idx1];
    cudarrows::Chunk &chunkB = chunks[idx2];
    cudarrows::chunkCoord adjacentChunksCoords[8][2] = {
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
    uint16_t savedChunkCount = reader.readU16();
    cudarrows::ArrowInfo *d_arrows;
    cuda_assert(cudaMalloc(&d_arrows, sizeof(cudarrows::ArrowInfo) * 256));
    bool *d_success;
    cuda_assert(cudaMalloc(&d_success, sizeof(bool)));
    for (uint16_t i = 0; i < savedChunkCount; ++i) {
        int16_t savedChunkX = reader.readI16();
        int16_t savedChunkY = reader.readI16();
        int8_t negX = savedChunkX < 0 ? 1 : 0;
        int8_t negY = savedChunkY < 0 ? 1 : 0;
        cudarrows::chunkCoord chunkX = (savedChunkX + negX) / 2 - negX;
        cudarrows::chunkCoord chunkY = (savedChunkY + negY) / 2 - negY;
        cudarrows::localCoord offsetX = (savedChunkX - chunkX * 2) * 16;
        cudarrows::localCoord offsetY = (savedChunkY - chunkY * 2) * 16;
        cudarrows::ArrowInfo h_arrows[256];
        uint8_t arrowTypeCount = reader.readU8();
        for (uint8_t j = 0; j <= arrowTypeCount; ++j) {
            uint8_t type = reader.readU8();
            uint8_t arrowCount = reader.readU8();
            for (uint8_t k = 0; k <= arrowCount; ++k) {
                uint8_t position = reader.readU8();
                uint8_t arrowX = position & 0xF;
                uint8_t arrowY = position >> 4;
                uint8_t rotation = reader.readU8();
                cudarrows::ArrowInfo &arrow = h_arrows[arrowY * 16 + arrowX];
                arrow.type = (cudarrows::ArrowType)type;
                arrow.rotation = (cudarrows::ArrowRotation)(rotation & 0x3);
                arrow.flipped = rotation & 0x4;
                if (k == 255) break;
            }
            if (j == 255) break;
        }
        cuda_assert(cudaMemcpy(d_arrows, h_arrows, sizeof(h_arrows), cudaMemcpyHostToDevice));
        cuda_assert(cudaMemset(d_success, false, sizeof(bool)));
        ::fillChunk<<<chunkCount, dim3(16, 16)>>>(chunks, chunkX, chunkY, offsetX, offsetY, 16, d_arrows, d_success);
        cuda_assert(cudaDeviceSynchronize());
        bool success;
        cuda_assert(cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
        if (!success) {
            cudarrows::Chunk *oldChunks = chunks;
            cuda_assert(cudaMalloc(&chunks, sizeof(cudarrows::Chunk) * ++chunkCount));
            cuda_assert(cudaMemcpy(chunks, oldChunks, sizeof(cudarrows::Chunk) * (chunkCount - 1), cudaMemcpyDeviceToDevice));
            cuda_assert(cudaMemset(&chunks[chunkCount - 1], 0, sizeof(cudarrows::Chunk)));
            cuda_assert(cudaFree(oldChunks));
            ::initChunk<<<1, dim3(16, 16)>>>(chunks, chunkCount - 1, chunkX, chunkY, offsetX, offsetY, 16, d_arrows);
        }
    }
    cuda_assert(cudaFree(d_arrows));
    cuda_assert(cudaFree(d_success));

    unsigned int numBlocks = (chunkCount + 31) / 32;
    ::linkChunks<<<dim3(numBlocks, numBlocks), dim3(32, 32)>>>(chunks, chunkCount);
    cuda_assert(cudaDeviceSynchronize());
}

cudarrows::Map::~Map() {
    cuda_assert(cudaFree(chunks));
}

cudarrows::ArrowInfo cudarrows::Map::getArrow(cudarrows::globalCoord x, cudarrows::globalCoord y) {
    cudarrows::chunkCoord chunkX = arrowToChunk(x);
    cudarrows::chunkCoord chunkY = arrowToChunk(y);
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

void cudarrows::Map::sendInput(cudarrows::globalCoord x, cudarrows::globalCoord y, cudarrows::ArrowInput input) {
    cudarrows::chunkCoord chunkX = arrowToChunk(x);
    cudarrows::chunkCoord chunkY = arrowToChunk(y);
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

void cudarrows::Map::render(cudaSurfaceObject_t surface, cudarrows::globalCoord minX, cudarrows::globalCoord minY, cudarrows::globalCoord maxX, cudarrows::globalCoord maxY) {
    ::render<<<chunkCount, dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(surface, chunks, step, minX, minY, maxX, maxY);
}

cudarrows::chunkCoord cudarrows::Map::arrowToChunk(cudarrows::globalCoord x) {
    int8_t neg = x < 0 ? 1 : 0;
    return (x + neg) / CHUNK_SIZE - neg;
}