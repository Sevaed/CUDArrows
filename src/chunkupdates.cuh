#pragma once
#include "map.cuh"

__global__ void update(cudarrows::Chunk *chunks, uint8_t step, uint8_t nextStep);

__global__ void reset(cudarrows::Chunk *chunks, uint64_t seed);