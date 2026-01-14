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

	particle_set(size_t N);

	void draw_particle(float x, float y, float z, float radius);

	void customDraw();
};

