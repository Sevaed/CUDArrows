#pragma once
#include <inttypes.h>
#include "map.cuh"

__global__ void render(cudaSurfaceObject_t surf, const cudarrows::Chunk *chunks, uint8_t step, cudarrows::globalCoord minX, cudarrows::globalCoord minY, cudarrows::globalCoord maxX, cudarrows::globalCoord maxY);