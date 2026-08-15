#pragma once

#include "particle.h"
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

class Renderer {
public:
    Renderer(unsigned width, unsigned height, float event_horizon);

    void resize(unsigned width, unsigned height);
    void draw(const particle_set& particles);

private:
    void setup_projection();
    void draw_sphere(float radius, int slices, int stacks);
    void draw_particles(const particle_set& particles);

    unsigned width_;
    unsigned height_;
    float event_horizon_;
    float camera_distance_ = 700.0f;
};
