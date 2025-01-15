#pragma once
#include <string>
#include <inttypes.h>
#include <curand.h>
#include <curand_kernel.h>

#define CHUNK_SIZE 32

namespace cudarrows {
    typedef int32_t globalCoord;
    typedef int16_t chunkCoord;
    typedef int8_t localCoord;
    typedef uint16_t arrowIdx;

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

    struct __builtin_align__(8) ArrowInfo {
        ArrowType type = ArrowType::Void;
        ArrowRotation rotation : 2;
        bool flipped : 1;

        ArrowInfo() : rotation(ArrowRotation::North), flipped(false) {}
    };

    struct __builtin_align__(8) ArrowState {
        ArrowSignal signal = ArrowSignal::White;
        uint8_t signalCount = 0;
        bool blocked = false;
    };

    union __builtin_align__(8) ArrowInput {
        bool buttonPressed;
        curandState_t curandState;
    };

    struct __builtin_align__(8) Arrow : ArrowInfo {
        ArrowState state[2];
        ArrowInput input;
    };

    struct __builtin_align__(8) Chunk {
        cudarrows::chunkCoord x, y;
        Chunk *adjacentChunks[8] = { nullptr };
        Arrow arrows[CHUNK_SIZE * CHUNK_SIZE];

        Chunk(cudarrows::chunkCoord x, cudarrows::chunkCoord y) : x(x), y(y) {}

        Chunk() : Chunk(0, 0) {}
    };

    class Map {
    private:
        Chunk *chunks = nullptr;
        unsigned int chunkCount = 0;
        uint8_t step = 0, nextStep = 1;
        
    public:
        Map() {}

        Map(const std::string &save);

        ~Map();

        ArrowInfo getArrow(cudarrows::globalCoord x, cudarrows::globalCoord y);

        void sendInput(cudarrows::globalCoord x, cudarrows::globalCoord y, ArrowInput input);

        void reset(uint64_t seed);

        void update();

        void render(cudaSurfaceObject_t surface, cudarrows::globalCoord minX, cudarrows::globalCoord minY, cudarrows::globalCoord maxX, cudarrows::globalCoord maxY);

        static cudarrows::chunkCoord arrowToChunk(cudarrows::globalCoord x);
    };
};