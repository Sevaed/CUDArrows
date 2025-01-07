#pragma once
#include <inttypes.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CHUNK_SIZE 16

namespace cudarrows {    
    struct Arrow {
        uint8_t type = 0;
        uint8_t rotation = 0;
        bool flipped = false;
    };

    struct ArrowState {
        uint8_t signal = 0;
        uint8_t signalCount = 0;
    };

    struct Chunk {
        uint16_t x, y;
        Arrow arrows[CHUNK_SIZE * CHUNK_SIZE];
        ArrowState states[CHUNK_SIZE * CHUNK_SIZE][2];

        Chunk(uint16_t x, uint16_t y) : x(x), y(y) {}

        Chunk() : Chunk(0, 0) {}
    };

    class Map {
    private:
        thrust::device_vector<Chunk> chunks;
        
    public:
        Map() {}

        void load(const std::string &save);
        
        std::string save();

        const Chunk *getChunks() const { return thrust::raw_pointer_cast(chunks.data()); };

        const Chunk getChunk(uint16_t x, uint16_t y);

        void setChunk(uint16_t x, uint16_t y, Chunk chunk);

        const Arrow getArrow(uint32_t x, uint32_t y);

        void setArrow(uint32_t x, uint32_t y, Arrow arrow);
    };
};