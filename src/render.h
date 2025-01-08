#pragma once
#include <inttypes.h>
#include "map.h"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, uint32_t minX, uint32_t minY, uint32_t maxX, uint32_t maxY);