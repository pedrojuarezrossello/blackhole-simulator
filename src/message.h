#pragma once

#include <array>
#include "utils.h"
#include "message_queue.h"

// Forward declaration
template <size_t N>
struct message_schwarzschild;

template <size_t N>
struct message_kerr;

// Must be changed when changing the spacetime... kinda hate this I'm tempted to type erase it via dynamic polymorphism
extern message_queue<message_kerr<N>> data_queue;

enum particle_state  {
	in_orbit,
	schwarzschild_radius
};

template <size_t N>
struct message_schwarzschild {
	alignas(64) std::array<float, N> xs;
	alignas(64) std::array<float, N> ys;
	alignas(64) std::array<float, N> zs;
	alignas(64) std::array<particle_state, N> states;

	void print() {
		std::cout << "Message start: " << std::endl;
		for (auto n : xs)
			std::cout << n << " ";

		std::cout << std::endl;
		for (auto n : ys)
			std::cout << n << " ";

		std::cout << std::endl;
	}

	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, MFLOAT bm_ps, size_t idx) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);
		MFLOAT xs_ps = MUL(MUL(radii_ps, cos_phis_ps), sin_thetas_ps);
		MFLOAT ys_ps = MUL(MUL(radii_ps, sin_phis_ps), sin_thetas_ps);
		MFLOAT zs_ps = MUL(radii_ps, cos_thetas_ps);

		// Won't be immediately used...
		STREAM(&xs[idx], xs_ps);
		STREAM(&ys[idx], ys_ps);
		STREAM(&zs[idx], zs_ps);

		// Determine if the particle has fallen into the schwarzschild
		// radius and thus will be stay there forever
		MFLOAT schwarzschild_radius_ps = SUB(SET1(0.0f), bm_ps);
		auto mask = CMP(radii_ps, schwarzschild_radius_ps, _CMP_LE_OQ);

		// The mask is set for the lanes which have fallen past the Schwarzschild radius (status 1)
		// For the others, the status is 0 hence the zero-mask move
		MINT schwarzschild_radius_epi32 = SET1_EPI32(static_cast<int>(particle_state::schwarzschild_radius));
#ifdef __AVX2__
		MINT part_states_epi32 = MASKZ_MOV_EPI32(_mm256_cvtps_epi32(mask), schwarzschild_radius_epi32);
#else
		MINT part_states_epi32 = MASKZ_MOV_EPI32(mask, schwarzschild_radius_epi32);
#endif
		STREAM_EPI32((MINT*) &states[idx], part_states_epi32);
	}

	void send() {
	//	data_queue.push(*this);
	}
};

template <size_t N>
struct message_kerr {
	alignas(64) std::array<float, N> xs;
	alignas(64) std::array<float, N> ys;
	alignas(64) std::array<float, N> zs;

	void print() {
		std::cout << "Message start: " << std::endl;
		for (auto n : xs)
			std::cout << n << " ";

		std::cout << std::endl;
		for (auto n : ys)
			std::cout << n << " ";

		std::cout << std::endl;
	}

	// TODO fix coordinate transformation
	void convert_and_add(MFLOAT radii_ps, MFLOAT phis_ps, MFLOAT thetas_ps, MFLOAT a_ps, size_t idx) {
		MFLOAT cos_phis_ps = COS(phis_ps);
		MFLOAT sin_phis_ps = SIN(phis_ps);
		MFLOAT cos_thetas_ps = COS(thetas_ps);
		MFLOAT sin_thetas_ps = SIN(thetas_ps);

		MFLOAT a_squared_ps = MUL(a_ps, a_ps);
		MFLOAT pseudo_radius_ps = SQRT(FMADD(radii_ps, radii_ps, a_squared_ps));
		
		MFLOAT xs_ps = MUL(MUL(pseudo_radius_ps, cos_phis_ps), sin_thetas_ps);
		MFLOAT ys_ps = MUL(MUL(pseudo_radius_ps, sin_phis_ps), sin_thetas_ps);
		MFLOAT zs_ps = MUL(pseudo_radius_ps, cos_thetas_ps);

		// Won't be immediately used...
		STORE(&xs[idx], xs_ps);
		STORE(&ys[idx], ys_ps);
		STORE(&zs[idx], zs_ps);
	}

	void send() {
		data_queue.push(*this);
	}
};
