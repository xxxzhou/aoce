#include "cudatest.h"

extern "C" void testFastGuided();
extern "C" void test10();

int main() {
    int intBufSize;
    nppsIntegralGetBufferSize_32s(1, &intBufSize);
    testFastGuided();

    return 0;
}