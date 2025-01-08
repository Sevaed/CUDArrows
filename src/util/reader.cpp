#include <stdexcept>
#include "util/reader.h"

util::Reader::Reader(std::string buffer) {
    this->buffer = (uint8_t *)buffer.c_str();
    this->end = this->buffer + buffer.size();
}

uint8_t util::Reader::read8() {
    if (this->buffer >= this->end)
        throw std::out_of_range("failed to read from buffer");
    return *(this->buffer++);
}

uint16_t util::Reader::read16() {
    return (uint16_t)read8() | ((uint16_t)read8() << 8);
}