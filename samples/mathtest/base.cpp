#include "base.hpp"

#include <iostream>
int32_t add(int32_t a, int32_t b) { return a + b; }

int main() {
    float x = pow(4, -1);
    float xx = pow(4, -0.5);
    float xxx = pow(16, 0.25);
    float xxxx = pow(16, -0.25);
    int y0 = 17;
    int x0 = 17;
    for (int i = 0, y = y0; i < 4 && y < 256; ++i, y += 16) {
        for (int j = 0, x = x0; j < 4 && x < 256; ++j, x += 16) {
            std::cout << "j:" << j << " x:" << x << " i:" << i << " y:" << y
                      << std::endl;
        }
    }
    uint32_t yy = 0;
    yy += 1;
    yy += (1 << 8);
    yy += (1 << 8);
    yy += (1 << 8);
    yy += (1 << 16);
    yy += (1 << 16);
    yy += (1 << 24);

    uint32_t r = (yy & 0x000000FF);
    uint32_t g = ((yy & 0x0000FF00) >> 8);
    uint32_t b = ((yy & 0x00FF0000) >> 16);
    uint32_t a = ((yy & 0xFF000000) >> 24);
    char c;
    std::cin >> c;
}