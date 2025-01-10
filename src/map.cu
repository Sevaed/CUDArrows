#include <string>
#include <stdexcept>
#include "base64/base64.h"
#include "util/reader.h"
#include "map.h"

namespace cudarrows {
    struct has_position {
        int16_t x, y;

        has_position(int16_t x, int16_t y) : x(x), y(y) {}

        __host__ __device__ bool operator()(cudarrows::Chunk chunk) {
            return chunk.x == x && chunk.y == y;
        }
    };
};

void cudarrows::Map::load(const std::string &save) {
    std::string buf = base64_decode(save);
    if (buf.size() < 4) return;
    util::Reader reader(buf);
    if (reader.read16() != 0)
        throw std::invalid_argument("Unsupported save version");
    uint16_t chunkCount = reader.read16();
    for (uint16_t i = 0; i < chunkCount; ++i) {
        int16_t chunkX = reader.read16();
        int16_t chunkY = reader.read16();
        uint8_t arrowTypeCount = reader.read8();
        cudarrows::Chunk chunk = getChunk(chunkX, chunkY);
        for (uint8_t j = 0; j <= arrowTypeCount; ++j) {
            uint8_t type = reader.read8();
            uint8_t arrowCount = reader.read8();
            for (uint8_t k = 0; k <= arrowCount; ++k) {
                uint8_t position = reader.read8();
                uint8_t arrowX = position & 0xF;
                uint8_t arrowY = position >> 4;
                uint8_t rotation = reader.read8();
                chunk.arrows[arrowY * CHUNK_SIZE + arrowX] = { (cudarrows::ArrowType)type, (cudarrows::ArrowRotation)(rotation & 0x3), (bool)(rotation & 0x4) };
                if (k == 255) break;
            }
            if (j == 255) break;
        }
        setChunk(chunk);
    }
}

const cudarrows::Chunk cudarrows::Map::getChunk(int16_t x, int16_t y) {
    thrust::device_vector<cudarrows::Chunk>::iterator iter = thrust::find_if(chunks.begin(), chunks.end(), cudarrows::has_position(x, y));
    return iter == chunks.end() ? cudarrows::Chunk(x, y) : iter[0];
}

void cudarrows::Map::setChunk(cudarrows::Chunk chunk) {
    thrust::device_vector<cudarrows::Chunk>::iterator iter = thrust::find_if(chunks.begin(), chunks.end(), cudarrows::has_position(chunk.x, chunk.y));
    if (iter != chunks.end())
        iter[0] = chunk;
    else {
        int16_t neighbours[][2] = {
            { chunk.x,     chunk.y - 1 },
            { chunk.x + 1, chunk.y - 1 },
            { chunk.x + 1, chunk.y     },
            { chunk.x + 1, chunk.y + 1 },
            { chunk.x,     chunk.y + 1 },
            { chunk.x - 1, chunk.y + 1 },
            { chunk.x - 1, chunk.y     },
            { chunk.x - 1, chunk.y - 1 }
        };
        for (uint8_t i = 0; i < sizeof(neighbours) / sizeof(neighbours[0]); ++i) {
            iter = thrust::find_if(chunks.begin(), chunks.end(), cudarrows::has_position(neighbours[i][0], neighbours[i][1]));
            if (iter != chunks.end()) {
                chunk.adjacentChunks[i] = thrust::distance(chunks.begin(), iter) + 1;
                size_t offset = chunks.size() + 1;
                cudarrows::Chunk *adjacentChunk = thrust::raw_pointer_cast(&iter[0]);
                cudaMemcpy(&adjacentChunk->adjacentChunks[(i + 4) % 8], &offset, sizeof(offset), cudaMemcpyHostToDevice);
            }
        }
        chunks.push_back(chunk);
    }
}

const cudarrows::Arrow cudarrows::Map::getArrow(int32_t x, int32_t y) {
    int16_t chunkX = x / CHUNK_SIZE;
    int16_t chunkY = y / CHUNK_SIZE;
    return getChunk(chunkX, chunkY).arrows[(y - chunkY) * CHUNK_SIZE + (x - chunkX)];
}

void cudarrows::Map::setArrow(int32_t x, int32_t y, cudarrows::Arrow arrow) {
    int16_t chunkX = x / CHUNK_SIZE;
    int16_t chunkY = y / CHUNK_SIZE;
    cudarrows::Chunk chunk = getChunk(chunkX, chunkY); // TODO: Change this to cudaMemcpy
    chunk.arrows[(y - chunkY) * CHUNK_SIZE + (x - chunkX)] = arrow;
    setChunk(chunk);
}