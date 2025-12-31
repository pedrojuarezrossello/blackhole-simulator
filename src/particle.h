#pragma once

#include "ofMain.h"
#include "message.h"
#include "utils.h"
#include <vector>

struct particle {
	particle() = default;
	particle(ofVec3f x)
		: pos(x) { }
	void draw();
	ofVec3f pos;
	ofPolyline trail;
	particle_state state = particle_state::in_orbit;
};

struct particle_set : public ofNode {
	void customDraw() { std::ranges::for_each(particles, &particle::draw); }

	// objects
	std::vector<particle> particles;
};

