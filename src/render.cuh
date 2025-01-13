#pragma once
#include <inttypes.h>
#include "map.cuh"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, int32_t minX, int32_t minY, int32_t maxX, int32_t maxY);