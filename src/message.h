#pragma once

#include <vector>
#include "utils.h"
#include "message_queue.h"

// Forward declaration
struct message;

extern message_queue<message> data_queue;

enum particle_state  {
	in_orbit,
	event_horizon
};

struct message {
	ALIGN std::vector<float> xs;
	ALIGN std::vector<float> ys;
	ALIGN std::vector<float> zs;
	ALIGN std::vector<particle_state> states;

	message() = default;

	message(size_t size)
		: xs(std::vector<float>(size))
		, ys(std::vector<float>(size))
		, zs(std::vector<float>(size))
		, states(std::vector<particle_state>(size)) {}

	void print() {
		std::cout << "Message start: " << std::endl;
		for (auto n : xs)
			std::cout << n << " ";

		std::cout << std::endl;
		for (auto n : ys)
			std::cout << n << " ";

		std::cout << std::endl;
	}

	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, MFLOAT bm_ps, size_t idx, schwarzschild _) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);
		MFLOAT xs_ps = MUL(MUL(radii_ps, cos_phis_ps), sin_thetas_ps);
		MFLOAT ys_ps = MUL(MUL(radii_ps, sin_phis_ps), sin_thetas_ps);
		MFLOAT zs_ps = MUL(radii_ps, cos_thetas_ps);

		STORE(&xs[idx], xs_ps);
		STORE(&ys[idx], ys_ps);
		STORE(&zs[idx], zs_ps);

		// Determine if the particle has fallen into the schwarzschild
		// radius and thus will be stay there forever
		MFLOAT schwarzschild_radius_ps = SUB(SET1(0.0f), bm_ps);
		auto mask = CMP(radii_ps, schwarzschild_radius_ps, _CMP_LE_OQ);

		// The mask is set for the lanes which have fallen past the Schwarzschild radius (status 1)
		// For the others, the status is 0 hence the zero-mask move
		MFLOAT event_horizon_ps = SET1(1.0f);
#ifdef __AVX2__
		MFLOAT part_states_ps = _mm256_blendv_ps(SETZERO, event_horizon_ps, mask);
		MINT part_states_epi32 = _mm256_cvtps_epi32(part_states_ps);
#else
		MINT part_states_epi32 = MASKZ_MOV_EPI32(mask, schwarzschild_radius_epi32);
#endif
		STORE_EPI32((MINT*) &states[idx], part_states_epi32);
	}

	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, MFLOAT a_ps, MFLOAT step_ps, size_t idx, kerr _) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);

		MFLOAT a_squared_ps = MUL(a_ps, a_ps);
		MFLOAT pseudo_radius_ps = SQRT(FMADD(radii_ps, radii_ps, a_squared_ps));

		MFLOAT xs_ps = MUL(MUL(pseudo_radius_ps, cos_phis_ps), sin_thetas_ps);
		MFLOAT ys_ps = MUL(MUL(pseudo_radius_ps, sin_phis_ps), sin_thetas_ps);
		MFLOAT zs_ps = MUL(pseudo_radius_ps, cos_thetas_ps);

		//If step_ps is zero, then we know that that particle has fallen into the black hole
		auto mask = CMP(step_ps, SETZERO, _CMP_EQ_OQ);
		MFLOAT event_horizon_ps = SET1(1.0f);
#ifdef __AVX2__
		MFLOAT part_states_ps = _mm256_blendv_ps(SETZERO, event_horizon_ps, mask);
		MINT part_states_epi32 = _mm256_cvtps_epi32(part_states_ps);
#else
		MINT part_states_epi32 = MASKZ_MOV_EPI32(mask, event_horizon_epi32);
#endif
		STORE_EPI32((MINT *)&states[idx], part_states_epi32);

		STORE(&xs[idx], xs_ps);
		STORE(&ys[idx], ys_ps);
		STORE(&zs[idx], zs_ps);
	}

	void convert_and_add(float radius, float phi, float theta, float bm, size_t idx, schwarzschild _) {
		xs[idx] = radius * std::cos(phi) * std::sin(theta);
		ys[idx] = radius * std::sin(phi) * std::sin(theta);
		zs[idx] = radius * std::cos(theta);
		states[idx] = radius > 2.0f * bm ? particle_state::in_orbit : particle_state::event_horizon;
	}

	void convert_and_add(float radius, float phi, float theta, float a, float step, size_t idx, kerr _) {
		float factor = std::sqrtf(radius * radius + a * a);
		xs[idx] = factor * std::cos(phi) * std::sin(theta);
		ys[idx] = factor * std::sin(phi) * std::sin(theta);
		zs[idx] = factor * std::cos(theta);
		states[idx] = step ? particle_state::in_orbit : particle_state::event_horizon;
	}

	void send() {
		data_queue.push(*this);
	}
};
