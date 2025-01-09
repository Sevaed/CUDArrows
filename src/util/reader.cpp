#include <stdexcept>
#include "util/reader.h"

uint8_t util::Reader::read8() {
    if (this->offset >= this->buffer.size())
        throw std::out_of_range("Failed to read from buffer");
    return this->buffer[this->offset++];
}

uint16_t util::Reader::read16() {
    return (uint16_t)read8() | ((uint16_t)read8() << 8);
}