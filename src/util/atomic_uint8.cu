#include "util/atomic_uint8.h"

// https://stackoverflow.com/a/59329536/16475499
__device__ unsigned char atomicAdd(unsigned char* address, unsigned char val) {
    size_t long_address_modulo = (size_t) address & 3;
    auto* base_address = (unsigned int *)(address - long_address_modulo);

    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;

    long_old = *base_address;

    do {
        long_assumed = long_old;
        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}