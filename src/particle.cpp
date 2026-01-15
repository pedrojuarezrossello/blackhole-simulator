#include "particle.h"

particle_set::particle_set(size_t N, float a, kerr _)
	: x(std::vector<float>(N))
	, y(std::vector<float>(N))
	, z(std::vector<float>(N))
	, rad(std::vector<float>(N))
	, states(std::vector<particle_state>(N))
	, ev_hor(scale_factor * (1+std::sqrtf(1-a*a))) {}

particle_set::particle_set(size_t N, float bm, schwarzschild _)
	: x(std::vector<float>(N))
	, y(std::vector<float>(N))
	, z(std::vector<float>(N))
	, rad(std::vector<float>(N))
	, states(std::vector<particle_state>(N))
	, ev_hor(scale_factor*2.0f*bm) { }

void particle_set::draw_particle(float x, float y, float z, float radius, particle_state state) {
	if (state == particle_state::event_horizon) return;

	float threshold_red = 3.0f * ev_hor;
	float threshold_green = 5.0f * ev_hor;
	float threshold_blue = 6.0f * ev_hor;

	auto get_red = [threshold_green, threshold_red](float r) {
		return (threshold_green - r) / (threshold_green - threshold_red);
	};

	auto get_green = [threshold_blue, threshold_green, threshold_red, a = this->ev_hor](float r) {
		return r < threshold_green ? (r - a) / (threshold_red - a) : (threshold_blue - r) / (threshold_blue - threshold_green);
	};

	auto get_blue = [threshold_green, threshold_blue](float r) {
		return (r - threshold_green) / (threshold_blue - threshold_green);
	};

	ofFloatColor c(get_red(radius), get_green(radius), get_blue(radius));
	ofSetColor(c);
	ofDrawSphere(x, y, z, 8.0f);
}

void particle_set::customDraw() {
	for (int i = 0; i < x.size(); ++i)
		draw_particle(x[i] * scale_factor, y[i] * scale_factor, z[i] * scale_factor, rad[i] * scale_factor, states[i]);
}


