#pragma once
#include <string>
#include <inttypes.h>

namespace util {
    class Reader {
    private:
        uint8_t *buffer;
        uint8_t *end;

    public:
        Reader(std::string buffer);

        uint8_t read8();

        uint16_t read16();
    };
};