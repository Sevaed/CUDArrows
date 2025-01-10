#pragma once
#include "map.h"

__global__ void update(cudarrows::Chunk *chunks, uint8_t step, uint8_t nextStep);

__global__ void reset(cudarrows::Chunk *chunks);