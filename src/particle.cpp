#include "particle.h"

particle_set::particle_set(size_t N)
	: x(std::vector<float>(N))
	, y(std::vector<float>(N))
	, z(std::vector<float>(N))
	, rad(std::vector<float>(N)) { }

void particle_set::draw_particle(float x, float y, float z, float radius) {
	float a = 1.97f * 30.0f;
	float threshold_red = 3.0f * a;
	float threshold_green = 5.0f * a;
	float threshold_blue = 7.0f * a;

	auto get_red = [threshold_green, threshold_red](float r) {
		return (threshold_green - r) / (threshold_green - threshold_red);
	};

	auto get_green = [threshold_blue, threshold_green, threshold_red, a](float r) {
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
		draw_particle(x[i], y[i], z[i], rad[i]);
}


