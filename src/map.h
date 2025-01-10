#pragma once
#include <inttypes.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CHUNK_SIZE 16

namespace cudarrows {
    enum class ArrowType : uint8_t {
        Void,
        Arrow,
        SourceBlock,
        Blocker,
        DelayArrow,
        SignalDetector,
        SplitterUpDown,
        SplitterUpRight,
        SplitterUpLeftRight,
        PulseGenerator,
        BlueArrow,
        DiagonalArrow,
        BlueSplitterUpUp,
        BlueSplitterUpRight,
        BlueSplitterUpDiagonal,
        NotGate,
        AndGate,
        XorGate,
        Latch,
        Flipflop,
        Randomizer,
        Button,
        Source,
        Target,
        DirectionalButton
    };

    enum class ArrowRotation : uint8_t {
        North,
        East,
        South,
        West
    };

    enum class ArrowSignal : uint8_t {
        White,
        Red,
        Blue,
        Yellow,
        Green,
        Orange,
        Magenta
    };

    struct ArrowState {
        ArrowSignal signal = ArrowSignal::White;
        uint8_t signalCount = 0;
        bool blocked = false;
    };

    struct Arrow {
        ArrowType type = ArrowType::Void;
        ArrowRotation rotation = ArrowRotation::North;
        bool flipped = false;
        ArrowState state[2];
    };

    struct Chunk {
        int16_t x, y;
        size_t adjacentChunks[8] = { 0 };
        Arrow arrows[CHUNK_SIZE * CHUNK_SIZE];

        Chunk(int16_t x, int16_t y) : x(x), y(y) {}

        Chunk() : Chunk(0, 0) {}
    };

    class Map {
    private:
        thrust::device_vector<Chunk> chunks;
        
    public:
        Map() {}

        void load(const std::string &save);

        const Chunk *getChunks() const { return thrust::raw_pointer_cast(chunks.data()); };

        size_t countChunks() const { return chunks.size(); };

        const Chunk getChunk(int16_t x, int16_t y);

        void setChunk(Chunk chunk);

        const Arrow getArrow(int32_t x, int32_t y);

        void setArrow(int32_t x, int32_t y, Arrow arrow);
    };
};