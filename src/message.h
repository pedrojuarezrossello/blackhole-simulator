#pragma once

#include <vector>
#include "utils.h"
#include "message_queue.h"

constexpr float scale_factor = 30.0f;
const MFLOAT scale_factor_ps = SET1(scale_factor);

// Forward declaration
struct message;

extern message_queue<message> data_queue;

struct message {
	ALIGN std::vector<float> xs;
	ALIGN std::vector<float> ys;
	ALIGN std::vector<float> zs;
	ALIGN std::vector<float> radii;
	
	message() = default;

	message(size_t size)
		: xs(std::vector<float>(size))
		, ys(std::vector<float>(size))
		, zs(std::vector<float>(size))
		, radii(std::vector<float>(size)) {}

	void print() {
		std::cout << "Message start: " << std::endl;
		for (auto n : radii)
			std::cout << n << " ";

		/* std::cout << std::endl;
		for (auto n : ys)
			std::cout << n << " ";

		std::cout << std::endl;*/
	}

	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, size_t idx) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);
		MFLOAT xs_ps = MUL(MUL(MUL(radii_ps, cos_phis_ps), sin_thetas_ps), scale_factor_ps);
		MFLOAT ys_ps = MUL(MUL(MUL(radii_ps, sin_phis_ps), sin_thetas_ps), scale_factor_ps);
		MFLOAT zs_ps = MUL(MUL(radii_ps, cos_thetas_ps), scale_factor_ps);

		STORE(&xs[idx], xs_ps);
		STORE(&ys[idx], ys_ps);
		STORE(&zs[idx], zs_ps);
		STORE(&radii[idx], MUL(radii_ps, scale_factor_ps));
	}

	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, MFLOAT a_ps, size_t idx) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);

		MFLOAT a_squared_ps = MUL(a_ps, a_ps);
		MFLOAT pseudo_radius_ps = SQRT(FMADD(radii_ps, radii_ps, a_squared_ps));

		MFLOAT xs_ps = MUL(MUL(MUL(pseudo_radius_ps, cos_phis_ps), sin_thetas_ps), scale_factor_ps);
		MFLOAT ys_ps = MUL(MUL(MUL(pseudo_radius_ps, sin_phis_ps), sin_thetas_ps), scale_factor_ps);
		MFLOAT zs_ps = MUL(MUL(pseudo_radius_ps, cos_thetas_ps), scale_factor_ps);

		STORE(&xs[idx], xs_ps);
		STORE(&ys[idx], ys_ps);
		STORE(&zs[idx], zs_ps);
		STORE(&radii[idx], MUL(radii_ps, scale_factor_ps));
	}

	void convert_and_add(float radius, float phi, float theta, size_t idx) {
		xs[idx] = scale_factor * radius * std::cos(phi) * std::sin(theta);
		ys[idx] = scale_factor * radius * std::sin(phi) * std::sin(theta);
		zs[idx] = scale_factor * radius * std::cos(theta);
		radii[idx] = scale_factor * radius;
	}

	void convert_and_add(float radius, float phi, float theta, float a, size_t idx) {
		float factor = std::sqrtf(radius * radius + a * a);
		xs[idx] = scale_factor * factor * std::cos(phi) * std::sin(theta);
		ys[idx] = scale_factor * factor * std::sin(phi) * std::sin(theta);
		zs[idx] = scale_factor * factor * std::cos(theta);
		radii[idx] = scale_factor * radius;
	}

	void send() {
		data_queue.push(std::move(*this));
	}
};
