#include "chunkupdates.h"

__device__ cudarrows::Arrow *getArrow(cudarrows::Chunk *chunks, cudarrows::Chunk &chunk, cudarrows::Arrow &arrow, uint3 pos, int8_t dx, int8_t dy) {
    if (arrow.flipped)
        dx = -dx;
    int16_t x = pos.x;
    int16_t y = pos.y;
    switch (arrow.rotation) {
        case cudarrows::ArrowRotation::North:
            y += dy;
            x += dx;
            break;
        case cudarrows::ArrowRotation::East:
            x -= dy;
            y += dx;
            break;
        case cudarrows::ArrowRotation::South:
            y -= dy;
            x -= dx;
            break;
        case cudarrows::ArrowRotation::West:
            x += dy;
            y -= dx;
            break;
    }
    cudarrows::Chunk *targetChunk = &chunk;
    if (x >= CHUNK_SIZE) {
        if (y >= CHUNK_SIZE) {
            targetChunk = chunk.adjacentChunks[3] == 0 ? nullptr : chunks + chunk.adjacentChunks[3];
            x -= CHUNK_SIZE;
            y -= CHUNK_SIZE;
      }  else if (y < 0) {
            targetChunk = chunk.adjacentChunks[1] == 0 ? nullptr : chunks + chunk.adjacentChunks[1] - 1;
            x -= CHUNK_SIZE;
            y += CHUNK_SIZE;
        } else {
            targetChunk = chunk.adjacentChunks[2] == 0 ? nullptr : chunks + chunk.adjacentChunks[2] - 1;
            x -= CHUNK_SIZE;
        }
    } else if (x < 0) {
        if (y < 0) {
            targetChunk = chunk.adjacentChunks[7] == 0 ? nullptr : chunks + chunk.adjacentChunks[7] - 1;
            x += CHUNK_SIZE;
            y += CHUNK_SIZE;
        } else if (y >= CHUNK_SIZE) {
            targetChunk = chunk.adjacentChunks[5] == 0 ? nullptr : chunks + chunk.adjacentChunks[5] - 1;
            x += CHUNK_SIZE;
            y -= CHUNK_SIZE;
        } else {
            targetChunk = chunk.adjacentChunks[6] == 0 ? nullptr : chunks + chunk.adjacentChunks[6] - 1;
            x += CHUNK_SIZE;
        }
    } else if (y < 0) {
        targetChunk = chunk.adjacentChunks[0] == 0 ? nullptr : chunks + chunk.adjacentChunks[0] - 1;
        y += CHUNK_SIZE;
    } else if (y >= CHUNK_SIZE) {
        targetChunk = chunk.adjacentChunks[4] == 0 ? nullptr : chunks + chunk.adjacentChunks[4] - 1;
        y -= CHUNK_SIZE;
    }
    return targetChunk == nullptr ? nullptr : &targetChunk->arrows[y * CHUNK_SIZE + x];
}

__device__ void sendSignal(cudarrows::Arrow *arrow, uint8_t step) {
    if (arrow && arrow->type != cudarrows::ArrowType::Void)
        ++arrow->state[step].signalCount;
}

__device__ void blockSignal(cudarrows::Arrow *arrow, uint8_t step) {
    if (arrow && arrow->type != cudarrows::ArrowType::Void)
        arrow->state[step].blocked = true;
}

__global__ void update(cudarrows::Chunk *chunks, uint8_t step, uint8_t nextStep) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    cudarrows::Arrow &arrow = chunk.arrows[idx];
    cudarrows::ArrowState &state = arrow.state[step];
    cudarrows::ArrowState &prevState = arrow.state[nextStep];
    if (state.blocked)
        state.signal = cudarrows::ArrowSignal::White;
    else
        switch (arrow.type) {
            case cudarrows::ArrowType::Arrow:
            case cudarrows::ArrowType::Blocker:
            case cudarrows::ArrowType::SplitterUpDown:
            case cudarrows::ArrowType::SplitterUpRight:
            case cudarrows::ArrowType::SplitterUpLeftRight:
            case cudarrows::ArrowType::Source:
            case cudarrows::ArrowType::Target:
                state.signal = state.signalCount > 0 ? cudarrows::ArrowSignal::Red : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::SourceBlock:
                state.signal = cudarrows::ArrowSignal::Red;
                break;
            case cudarrows::ArrowType::DelayArrow:
                if (state.signalCount > 0)
                    state.signal = prevState.signal == cudarrows::ArrowSignal::Red ? cudarrows::ArrowSignal::Red : cudarrows::ArrowSignal::Blue;
                else
                    state.signal = prevState.signal == cudarrows::ArrowSignal::Blue ? cudarrows::ArrowSignal::Red : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::SignalDetector: {
                cudarrows::Arrow *arrowBehind = getArrow(chunks, chunk, arrow, threadIdx, 0, -1);
                state.signal =
                    arrowBehind == nullptr || arrowBehind->state[nextStep].signal == cudarrows::ArrowSignal::White ?
                        cudarrows::ArrowSignal::White :
                        cudarrows::ArrowSignal::Red;
                break;   
            }
            case cudarrows::ArrowType::PulseGenerator:
                state.signal = prevState.signal == cudarrows::ArrowSignal::Red ? cudarrows::ArrowSignal::Blue : cudarrows::ArrowSignal::Red;
                break;
            case cudarrows::ArrowType::BlueArrow:
            case cudarrows::ArrowType::DiagonalArrow:
            case cudarrows::ArrowType::BlueSplitterUpUp:
            case cudarrows::ArrowType::BlueSplitterUpRight:
            case cudarrows::ArrowType::BlueSplitterUpDiagonal:
                state.signal = state.signalCount > 0 ? cudarrows::ArrowSignal::Blue : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::NotGate:
                state.signal = state.signalCount == 0 ? cudarrows::ArrowSignal::Yellow : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::AndGate:
                state.signal = state.signalCount >= 2 ? cudarrows::ArrowSignal::Yellow : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::XorGate:
                state.signal = state.signalCount % 2 == 1 ? cudarrows::ArrowSignal::Yellow : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::Latch:
                if (state.signalCount > 0)
                    state.signal = state.signalCount >= 2 ? cudarrows::ArrowSignal::Yellow : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::Flipflop:
                if (state.signalCount > 0)
                    state.signal = (cudarrows::ArrowSignal)((uint8_t)cudarrows::ArrowSignal::Yellow - (uint8_t)prevState.signal);
                break;
            case cudarrows::ArrowType::DirectionalButton:
                state.signal = state.signalCount > 0 ? cudarrows::ArrowSignal::Orange : cudarrows::ArrowSignal::White;
                break;
        }
    switch (arrow.type) {
        case cudarrows::ArrowType::Arrow:
        case cudarrows::ArrowType::DelayArrow:
        case cudarrows::ArrowType::SignalDetector:
        case cudarrows::ArrowType::Source:
            if (prevState.signal == cudarrows::ArrowSignal::Red)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
            break;
        case cudarrows::ArrowType::SourceBlock:
        case cudarrows::ArrowType::PulseGenerator:
            if (prevState.signal == cudarrows::ArrowSignal::Red) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  1,  0), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  0,  1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, -1,  0), nextStep);
            }
            break;
        case cudarrows::ArrowType::Blocker:
            if (prevState.signal == cudarrows::ArrowSignal::Red)
                blockSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
            break;
        case cudarrows::ArrowType::SplitterUpDown:
            if (prevState.signal == cudarrows::ArrowSignal::Red) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0,  1), nextStep);
            }
            break;
        case cudarrows::ArrowType::SplitterUpRight:
            if (prevState.signal == cudarrows::ArrowSignal::Red) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 1,  0), nextStep);
            }
            break;
        case cudarrows::ArrowType::SplitterUpLeftRight:
            if (prevState.signal == cudarrows::ArrowSignal::Red) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, -1,  0), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  1,  0), nextStep);
            }
            break;
        case cudarrows::ArrowType::BlueArrow:
            if (prevState.signal == cudarrows::ArrowSignal::Blue)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -2), nextStep);
            break;
        case cudarrows::ArrowType::DiagonalArrow:
            if (prevState.signal == cudarrows::ArrowSignal::Blue)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 1, -1), nextStep);
            break;
        case cudarrows::ArrowType::BlueSplitterUpUp:
            if (prevState.signal == cudarrows::ArrowSignal::Blue) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -2), nextStep);
            }
            break;
        case cudarrows::ArrowType::BlueSplitterUpRight:
            if (prevState.signal == cudarrows::ArrowSignal::Blue) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -2), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 1,  0), nextStep);
            }
            break;
        case cudarrows::ArrowType::BlueSplitterUpDiagonal:
            if (prevState.signal == cudarrows::ArrowSignal::Blue) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 1, -1), nextStep);
            }
            break;
        case cudarrows::ArrowType::NotGate:
        case cudarrows::ArrowType::AndGate:
        case cudarrows::ArrowType::XorGate:
        case cudarrows::ArrowType::Latch:
        case cudarrows::ArrowType::Flipflop:
            if (prevState.signal == cudarrows::ArrowSignal::Yellow)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
            break;
        case cudarrows::ArrowType::Randomizer:
        case cudarrows::ArrowType::DirectionalButton:
            if (prevState.signal == cudarrows::ArrowSignal::Orange)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
            break;
        case cudarrows::ArrowType::Button:
            if (prevState.signal == cudarrows::ArrowSignal::Orange) {
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  0, -1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  1,  0), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx,  0,  1), nextStep);
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, -1,  0), nextStep);
            }
            break;
    }
    state.signalCount = 0;
    state.blocked = false;
}

__global__ void reset(cudarrows::Chunk *chunks) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    chunk.arrows[idx].state[blockIdx.y] = cudarrows::ArrowState();
}