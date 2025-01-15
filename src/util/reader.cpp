#include <stdexcept>
#include "util/reader.h"

uint8_t util::Reader::readU8() {
    if (this->offset >= this->buffer.size())
        throw std::out_of_range("Failed to read from buffer");
    return this->buffer[this->offset++];
}

uint16_t util::Reader::readU16() {
    return (uint16_t)readU8() | ((uint16_t)readU8() << 8);
}

int16_t util::Reader::readI16() {
    uint16_t u16 = readU16();
    int16_t i16 = int16_t(u16 & 0x7FFF);
    return u16 & 0x8000 ? -i16 : i16;
}