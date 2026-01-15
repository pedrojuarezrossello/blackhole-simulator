#pragma once

#include "ofMain.h"
#include "message.h"
#include "utils.h"
#include <vector>
#include <ranges>


struct particle_set : public ofNode {
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::vector<float> rad;
	std::vector<particle_state> states;
	float ev_hor;

	particle_set(size_t N, float a, kerr _);
	particle_set(size_t N, float bm, schwarzschild _);

	void draw_particle(float x, float y, float z, float radius, particle_state state);

	void customDraw();
};

