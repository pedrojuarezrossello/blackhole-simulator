#include "renderer.h"
#include "utils.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr float pi = 3.14159265358979323846f;

float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

void perspective(float fov_y_deg, float aspect, float z_near, float z_far) {
    const float f = 1.0f / std::tan(fov_y_deg * pi / 360.0f);
    const float m[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (z_far + z_near) / (z_near - z_far), -1,
        0, 0, (2.0f * z_far * z_near) / (z_near - z_far), 0
    };
    glMultMatrixf(m);
}
}

Renderer::Renderer(unsigned width, unsigned height, float event_horizon)
    : width_(width), height_(height), event_horizon_(event_horizon) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    const GLfloat light_position[] = {100.0f, 500.0f, 500.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glClearColor(0.f, 0.f, 0.f, 1.f);
    resize(width, height);
}

void Renderer::resize(unsigned width, unsigned height) {
    width_ = width;
    height_ = std::max(height, 1u);
    glViewport(0, 0, static_cast<GLsizei>(width_), static_cast<GLsizei>(height_));
}

void Renderer::setup_projection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    perspective(45.0f, static_cast<float>(width_) / static_cast<float>(height_), 1.0f, 5000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // This reproduces the useful part of the old ofEasyCam setup: a fixed
    // elevated view looking towards the origin.
    glTranslatef(0.0f, 0.0f, -camera_distance_);
    glRotatef(-60.0f, 1.0f, 0.0f, 0.0f);
}

void Renderer::draw_sphere(float radius, int slices, int stacks) {
    for (int stack = 0; stack < stacks; ++stack) {
        const float phi0 = pi * (-0.5f + static_cast<float>(stack) / stacks);
        const float phi1 = pi * (-0.5f + static_cast<float>(stack + 1) / stacks);
        const float z0 = std::sin(phi0);
        const float z1 = std::sin(phi1);
        const float r0 = std::cos(phi0);
        const float r1 = std::cos(phi1);

        glBegin(GL_QUAD_STRIP);
        for (int slice = 0; slice <= slices; ++slice) {
            const float theta = 2.0f * pi * static_cast<float>(slice) / slices;
            const float c = std::cos(theta);
            const float s = std::sin(theta);

            glNormal3f(r1 * c, r1 * s, z1);
            glVertex3f(radius * r1 * c, radius * r1 * s, radius * z1);
            glNormal3f(r0 * c, r0 * s, z0);
            glVertex3f(radius * r0 * c, radius * r0 * s, radius * z0);
        }
        glEnd();
    }
}

void Renderer::draw_particles(const particle_set& particles) {
    glDisable(GL_LIGHTING);
    glPointSize(8.0f);
    glBegin(GL_POINTS);

    const size_t n = std::min({particles.x.size(), particles.y.size(), particles.z.size(),
                               particles.rad.size(), particles.states.size()});
    const float threshold_red = 3.0f * particles.ev_hor;
    const float threshold_green = 5.0f * particles.ev_hor;
    const float threshold_blue = 6.0f * particles.ev_hor;

    for (size_t i = 0; i < n; ++i) {
        if (particles.states[i] == particle_state::event_horizon)
            continue;

        const float r = particles.rad[i] * scale_factor;
        const float red = clamp01((threshold_green - r) / (threshold_green - threshold_red));
        const float green = clamp01(r < threshold_green
            ? (r - particles.ev_hor) / (threshold_red - particles.ev_hor)
            : (threshold_blue - r) / (threshold_blue - threshold_green));
        const float blue = clamp01((r - threshold_green) / (threshold_blue - threshold_green));

        glColor3f(red, green, blue);
        glVertex3f(particles.x[i] * scale_factor,
                   particles.y[i] * scale_factor,
                   particles.z[i] * scale_factor);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void Renderer::draw(const particle_set& particles) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setup_projection();

    // Black hole.
    glColor3f(1.0f, 0.27f, 0.0f); // close to ofColor::orangeRed
    const GLfloat mat_shininess[] = {128.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
    draw_sphere(2.0f * scale_factor, 64, 32);

    draw_particles(particles);
}
