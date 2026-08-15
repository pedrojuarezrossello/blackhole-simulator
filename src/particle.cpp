#include "particle.h"
#include <cmath>
#include <utility>

particle_set::particle_set(size_t N, float a, kerr _)
    : x(N), y(N), z(N), rad(N), states(N),
      ev_hor(scale_factor * (1.0f + std::sqrt(1.0f - a * a))) {}

particle_set::particle_set(size_t N, float bm, schwarzschild _)
    : x(N), y(N), z(N), rad(N), states(N),
      ev_hor(scale_factor * 2.0f * bm) {}

void particle_set::update(std::vector<float> _x,
                          std::vector<float> _y,
                          std::vector<float> _z,
                          std::vector<float> _rad,
                          std::vector<particle_state> _states) {
    x = std::move(_x);
    y = std::move(_y);
    z = std::move(_z);
    rad = std::move(_rad);
    states = std::move(_states);
}
