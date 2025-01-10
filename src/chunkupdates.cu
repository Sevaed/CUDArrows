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
    if (arrow && arrow->type != 0)
        ++arrow->state[step].signalCount;
}

__device__ void blockSignal(cudarrows::Arrow *arrow, uint8_t step) {
    if (arrow && arrow->type != 0)
        arrow->state[step].blocked = true;
}

__global__ void update(cudarrows::Chunk *chunks, uint8_t step, uint8_t nextStep) {
    cudarrows::Chunk &chunk = chunks[blockIdx.x];
    int32_t x = chunk.x * CHUNK_SIZE + threadIdx.x;
    int32_t y = chunk.y * CHUNK_SIZE + threadIdx.y;
    uint8_t idx = threadIdx.y * CHUNK_SIZE + threadIdx.x;
    cudarrows::Arrow &arrow = chunk.arrows[idx];
    cudarrows::ArrowState &state = arrow.state[step];
    cudarrows::ArrowState &prevState = arrow.state[nextStep];
    if (state.blocked)
        state.signal = cudarrows::ArrowSignal::White;
    else
        switch (arrow.type) {
            case cudarrows::ArrowType::ArrowUp:
            case cudarrows::ArrowType::Blocker:
                state.signal = state.signalCount > 0 ? cudarrows::ArrowSignal::Red : cudarrows::ArrowSignal::White;
                break;
            case cudarrows::ArrowType::Source:
                state.signal = cudarrows::ArrowSignal::Red;
                break;
        }
    switch (arrow.type) {
        case cudarrows::ArrowType::ArrowUp:
            if (prevState.signal == cudarrows::ArrowSignal::Red)
                sendSignal(getArrow(chunks, chunk, arrow, threadIdx, 0, -1), nextStep);
            break;
        case cudarrows::ArrowType::Source:
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
    }
    state.signalCount = 0;
    state.blocked = false;
}