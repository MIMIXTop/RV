#pragma once
#include <array>

struct Point {
    std::array<double, 9> coords;

    int cluster_id = -1; 
};