#include "base.hpp"

#include <iostream>
int32_t add(int32_t a, int32_t b) { return a + b; }

void main() {
    float x = pow(4, -1);
    float xx = pow(4, -0.5);
    float xxx = pow(16, 0.25);
    float xxxx = pow(16, -0.25);
    int y0 = 17;
    int x0 = 17;
    for (int i = 0, y = y0; i < 4 && y < 256; ++i, y += 16) {
        for (int j = 0, x = x0; j < 4 && x < 256; ++j, x += 16) {
            std::cout << "j:" << j << " x:" << x << " i:" << i << " y:" << y << std::endl;
        }
    }
    char c;
    std::cin >> c;
}