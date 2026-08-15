#pragma once

#include "message.h"
#include "utils.h"
#include <cstddef>
#include <vector>

struct particle_set {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> rad;
    std::vector<particle_state> states;
    float ev_hor;

    particle_set(size_t N, float a, kerr _);
    particle_set(size_t N, float bm, schwarzschild _);

    void update(std::vector<float> _x,
                std::vector<float> _y,
                std::vector<float> _z,
                std::vector<float> _rad,
                std::vector<particle_state> _states);
};
