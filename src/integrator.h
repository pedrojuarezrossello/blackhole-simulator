#pragma once

#include <immintrin.h>
#include <array>
#include <thread>
#include <chrono>
#include <iostream>
#include "utils.h"

	/*
	We're trying to implement the Range-Kutta 4 method to solve the
	following ODE:

		(dr / dT)^2 = E^2 - V^2	and r(0) = r_0  (*)

	where E is the total energy of a test particle of unit mass, V is
	the potential energy of the test particle defined as

		V(r)^2 := (1 - 2 * M / r) * (1 + L^2 / r^2),

	where M is the mass of the black hole and L is the angular momentum.
	Note that E and L are conserved. Phi is then solved via

		dphi / dT = L / r^2   and phi(0) = phi_0,

	via the classical definition L := I * dphi / dT, and I being r^2 for
	a test particle of unit mass in circular motion.

	Note that we'll have to pick the sign of the square root in (*) according
	to the potential energy - if the energy is decreasing, then we pick the
	negative sign because the particle is falling inwards; otherwise, we pick
	the positive sign. 

	The Range-Kutta method with 4 stages can be used to solve an autonomous
	ODE (like the one above)

		dy / dt = f(y) and y(t_0) = y_0.

	Pick a step size h>0 and define

		y_n+1 = y_n + h/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
		t_n+1 = t_n + h;

	for n = 0, 1, 2... with

		k_1 = f(y_n)
		k_2 = f(y_n + h/2 * k_1)
		k_3 = f(y_n + h/2 * k_2)
		k_4 = f(y_n + h * k_3).

	Then y_n approximates y(t_n) with a total error of order O(h^4).
*/


constexpr float step = 0.5f;

class schwarzschild_integrator {
	// It's negative i.e. -2M !!
	MFLOAT _2black_hole_mass;

	ALIGN std::vector<float> angular_momenta;
	ALIGN std::vector<float> radii;
	ALIGN std::vector<float> phis;
	ALIGN std::vector<float> total_energies;
	ALIGN std::vector<float> directions;

	void _update_directions(MFLOAT* new_radii_ps, size_t idx) {
		// Work out for which lanes E^2-V^2 <= 0 
		MFLOAT energy_ps = LOAD(&total_energies[idx]);
		MFLOAT potential_minus_energy_ps = _rhs_radius_ode_squared(energy_ps, *new_radii_ps, idx);
		MFLOAT nonnegative_mask = CMP(potential_minus_energy_ps, SETZERO, _CMP_LE_OQ);

		// Swap the directions of these lanes
		MFLOAT direction_ps = LOAD(&directions[idx]);
		MFLOAT minus_one_ps = SET1(-1.0f);
		MFLOAT res = MASK_MUL(nonnegative_mask, direction_ps, minus_one_ps, direction_ps);
		MFLOAT adjusted_radius_ps = *new_radii_ps;
	
		//Update the corresponding radii to wiggle them away from the energy peak where E^2-V^2=0
		while (!TESTZ(nonnegative_mask, nonnegative_mask)) {
			MFLOAT step_ps = SET1(0.05f);
			// Move radius a bit further along the direction of direction_ps
			step_ps = MUL(step_ps, res);
			adjusted_radius_ps = MASK_ADD(nonnegative_mask, adjusted_radius_ps, adjusted_radius_ps, step_ps);
			potential_minus_energy_ps = _rhs_radius_ode_squared(energy_ps, adjusted_radius_ps, idx);
			nonnegative_mask = CMP(potential_minus_energy_ps, SETZERO, _CMP_LE_OQ);
		}
		
		STORE(&directions[idx], res);
		STORE(&radii[idx], adjusted_radius_ps);
		*new_radii_ps = adjusted_radius_ps;
	}

	MFLOAT _rhs_radius_ode_squared(MFLOAT total_energy, MFLOAT arg_for_potential, size_t idx) {
		// a^2 - b^2 = a*a - b^2
		MFLOAT potential = _potential_energy(arg_for_potential, idx);
		return FMSUB(total_energy, total_energy, MUL(potential, potential));
	}

	MFLOAT _next_step_radius(MFLOAT prev_radius, float step, size_t idx) {
		MFLOAT energy = LOAD(&total_energies[idx]);
		// k_1 = f(y_n)
		MFLOAT k1_squared_ps = _rhs_radius_ode_squared(energy, prev_radius, idx);
		MFLOAT k1_ps = SQRT(k1_squared_ps);

		// k_2 = f(y_n + h/2 * k_1)
		MFLOAT half_step_ps = SET1(step * 0.5f);
		MFLOAT arg_k2_ps = FMADD(half_step_ps, k1_ps, prev_radius);
		MFLOAT k2_squared_ps = _rhs_radius_ode_squared(energy, arg_k2_ps, idx);
		MFLOAT nonnegative_mask = CMP(k2_squared_ps, SETZERO, _CMP_LE_OQ);
		MFLOAT k2_ps = MASK_MOVE(k1_squared_ps, k2_squared_ps, nonnegative_mask);

		// k_3 = f(y_n + h/2 * k_2)
		MFLOAT arg_k3_ps = FMADD(half_step_ps, k2_ps, prev_radius);
		MFLOAT k3_squared_ps = _rhs_radius_ode_squared(energy, arg_k3_ps, idx);
		nonnegative_mask = CMP(k3_squared_ps, SETZERO, _CMP_LE_OQ);
		MFLOAT k3_ps = MASK_MOVE(k2_squared_ps, k3_squared_ps, nonnegative_mask);

		// k_4 = f(y_n + h * k_3)
		MFLOAT step_ps = SET1(step);
		MFLOAT arg_k4_ps = FMADD(step_ps, k3_ps, prev_radius);
		MFLOAT k4_squared_ps = _rhs_radius_ode_squared(energy, arg_k4_ps, idx);
		nonnegative_mask = CMP(k4_squared_ps, SETZERO, _CMP_LE_OQ);
		MFLOAT k4_ps = MASK_MOVE(k3_squared_ps, k4_squared_ps, nonnegative_mask);

		// Compute h/6 * (k_1 + 2k_2 + 2k_3 + k_4)
		MFLOAT k1_plus_k4_ps = ADD(k1_ps, k4_ps);
		MFLOAT k2_plus_k3_ps = ADD(k2_ps, k3_ps);
		MFLOAT sum_of_ks_ps = FMADD(SET1(2.0f), k2_plus_k3_ps, k1_plus_k4_ps);

		// Multiply by direction
		MFLOAT signed_sum_of_ks_ps = MUL(sum_of_ks_ps, LOAD(&directions[idx]));
		MFLOAT h_over_6_ps = SET1(step / 6.0f);

		// new_radius = prev_radius +- h/6 * (k_1 + 2k_2 + 2k_3 + k_4)
		return FMADD(h_over_6_ps, signed_sum_of_ks_ps, prev_radius);
	}

	MFLOAT _next_step_phi(MFLOAT radius, MFLOAT prev_phi, float step, size_t idx) {
		// new_phi = prev_phi + step * L / (rsin(theta))^2;
		MFLOAT radius_inv_ps = RCP(radius);
		MFLOAT radius_inv_squared_ps = MUL(radius_inv_ps, radius_inv_ps);
		MFLOAT step_ps = SET1(step);
		MFLOAT step_times_angular_momenta = MUL(step_ps, LOAD(&angular_momenta[idx]));
		return FMADD(step_times_angular_momenta, radius_inv_squared_ps, prev_phi);
	}

	MFLOAT _potential_energy(MFLOAT radius, size_t idx) {
		MFLOAT one_ps = SET1(1.0f);
		// Compute radius inverse with a built-in intrisic 
		MFLOAT radius_inv_ps = RCP(radius);
		MFLOAT radius_inv_squared_ps = MUL(radius_inv_ps, radius_inv_ps);
		// Note that _2black_hole_mass is negative
		MFLOAT first_term_ps = FMADD(_2black_hole_mass, radius_inv_ps, one_ps);
		MFLOAT angular_momenta_ps = LOAD(&angular_momenta[idx]);
		MFLOAT second_term_ps = FMADD(MUL(angular_momenta_ps, angular_momenta_ps), radius_inv_squared_ps, one_ps);
		return MUL(first_term_ps, second_term_ps);
	}
	std::vector<float> _initialise_energies(const std::vector<float>& radii) {
		ALIGN std::vector<float> energies(radii.size());
		size_t N = radii.size();
		for (size_t i = 0; i < N; i += subproblem_size) {
			MFLOAT initial_radii_ps = LOAD(&radii[i]);
			// Compute the potential energies for this chunk
			MFLOAT potential_energies_ps = _potential_energy(initial_radii_ps, i);
			STORE(&energies[i], potential_energies_ps);
		}
		return energies;
	}

	public:
		schwarzschild_integrator(float                _black_hole_mass,
			initial_particle_data<spacetime::schwarzschild> _initial_data)
			: _2black_hole_mass(SET1(-2 * _black_hole_mass))
			, angular_momenta(std::move(_initial_data.angular_momenta))
			, radii(std::move(_initial_data.initial_radii))
			, phis(std::move(_initial_data.initial_phis))
			, total_energies(_initialise_energies(radii))
			, directions(std::vector<float>(radii.size(), - 1.0f))
		{
			size_t N = radii.size();
			assert(angular_momenta.size() == N, "Angular momenta must have the same size as radii");
			assert(phis.size() == N, "Phi values must have the same size as radii");
			assert(_black_hole_mass > 0.0f, "Black hole mass must be positive");
		}

		MFLOAT next_radius(float step, size_t idx) {
			// Load and compute next radii
			MFLOAT prev_radius_ps = LOAD(&radii[idx]);
			MFLOAT next_radius_ps = _next_step_radius(prev_radius_ps, step, idx);

			// Store result
			STORE(&radii[idx], next_radius_ps);

			return next_radius_ps;
		}

		MFLOAT next_phi(float step, size_t idx) {
			// Load and compute next phis
			MFLOAT radius_ps = LOAD(&radii[idx]);
			MFLOAT prev_phi_ps = LOAD(&phis[idx]);
			MFLOAT next_phi_ps = _next_step_phi(radius_ps, prev_phi_ps, step, idx);

			// Store result
			STORE(&phis[idx], next_phi_ps);

			return next_phi_ps;
		}

		void print_radii() {
			size_t N = radii.size();
			std::cout << "Radii: ";
			for (size_t i = 0; i < N; ++i)
				std::cout << radii[i] << " ";
			std::cout << std::endl;
		}

		void send_data() {
			message_schwarzschild data(radii.size());
			size_t N = radii.size();
			for (size_t i = 0; i < N; i += subproblem_size) {
				MFLOAT radii_chunk = next_radius(step, i);
				_update_directions(&radii_chunk, i);
				MFLOAT phis_chunk = next_phi(step, i);
				MFLOAT thetas_chunk = SET1(3.141592f / 2.0f); 
				data.convert_and_add(radii_chunk, phis_chunk, thetas_chunk, _2black_hole_mass, i);
			}

			data.send();
		}

		void send_initial_data() {
			message_schwarzschild data(radii.size());
			size_t N = radii.size();
			for (size_t i = 0; i < N; i += subproblem_size) {
				MFLOAT radii_chunk = LOAD(&radii[i]);
				MFLOAT phis_chunk = LOAD(&phis[i]);
				MFLOAT thetas_chunk = SET1(3.141592f / 2.0f); 
				data.convert_and_add(radii_chunk, phis_chunk, thetas_chunk, _2black_hole_mass, i);
			}
			
			data.send();
		}

		void rock_n_roll() {
			send_initial_data();
			 for (;;) 
				send_data();
		}
};

/*
* Reference: https://www.aanda.org/articles/aa/pdf/2004/36/aa0814.pdf
*
* We use the following equations of motion for the geodesics of Kerr spacetime. First define
*	D = r^2 - 2Mr + a^2		and		S = r^2 + a^2cos^2(theta)
*	k = Q + L^2 + a^2(E^2-1),
* where M is the mass of the black hole, E is the particle's energy at infinity,
* Q is Carter's constant, L is the angular momentum of the particle in the phi
* direction and a is such that a/M defines the spin of the black hole. We'll assume that M = 1.
*
*	dphi / dT = (2arE + ((S-2r)L/sin^2(theta))) / SD
*	
*	dr / dT = D/S *p_r
*
*	dtheta / dT = p_theta / S
*
*	dp_r / dT = 1/SD * ((-r^2-a^2-k)(r-1)-rD+2r(r^2+a^2)E^2-2aEL)-2*p_r^2(r-1)/S
*
*	dp_theta / dT = (sin(theta) * cos(theta) / S)*(L^2/sin^4(theta)-a^2(E^2-1))
*
*  Here p stands for the linear momentum as usual. We'll solve them using the
*  Runge-Kutta-Fehlberg method.
*
*	The Butcher's tableau
*	1/4  |  1/4
*   3/8  |  3/32		9/32
*   12/13|  1932/2197	-7200/2197	7296/2197
*   1    |  439/216		-8			3680/513	-845/4104
*   1/2  |  -8/27		2			-3544/2565	1859/4104	-11/40
*	-------------------------------------------------------
*			25/216		0			1408/2565	2197/4104	-1/5		 (4th order)
*			16/135		0			6656/12825	28561/56430 -9/50	2/55 (5th order)
*
*	k_i = h*f(...) !!
*
*	The truncation error is
*
*	T = | sum_i=0_to_5 (c_hat(i) - c(i))*k_i |
*
*	For a given E, we set the new step to
*
*		h_new = 0.9*h*(E/T)^(1/4)
*
*	If T>E, we redo the same calculation with h_new as our step size. Else we move on.
*/

const MFLOAT tolerance_ps = SET1(0.05f);

const MFLOAT minus_two_ps = SET1(-2.0f);
const MFLOAT one_ps = SET1(1.0f);

// Step size calculation
const MFLOAT zero_point_nine_ps = SET1(0.9f);

// k2
const MFLOAT _1_4_ps = SET1(1.0f / 4.0f);

// k3
const MFLOAT _3_32_ps = SET1(3.0f / 32.0f);
const MFLOAT _9_32_ps = SET1(9.0f / 32.0f);

// k4
const MFLOAT _1932_2197_ps = SET1(1932.0f / 2197.0f);
const MFLOAT _minus_7200_2197_ps = SET1(-7200.0f / 2197.0f);
const MFLOAT _7296_2197_ps = SET1(7296.0f / 2197.0f);

// k5
const MFLOAT _439_216_ps = SET1(439.0f / 216.0f);
const MFLOAT _minus_8_ps = SET1(-8.0f);
const MFLOAT _3680_513_ps = SET1(3680.0f / 513.0f);
const MFLOAT _minus_845_4104_ps = SET1(-845.0f / 4104.0f);

// k6
const MFLOAT _minus_8_27_ps = SET1(-8.0f / 27.0f);
const MFLOAT two_ps = SET1(2.0f);
const MFLOAT _minus_3544_2565_ps = SET1(-3544.0f / 2565.0f);
const MFLOAT _1859_4104_ps = SET1(1858.0f / 4104.0f);
const MFLOAT _minus_11_40_ps = SET1(-11.0f / 40.0f);

// c order 4
const MFLOAT _25_216_ps = SET1(25.0f / 216.0f);
const MFLOAT _1408_2565_ps = SET1(1408.0f / 2565.0f);
const MFLOAT _2197_4104_ps = SET1(2197.0f / 4104.0f);
const MFLOAT _minus_1_5_ps = SET1(-1.0f / 5.0f);

// c order 5
const MFLOAT _16_135_ps = SET1(16.0f / 135.0f);
const MFLOAT _6656_12825_ps = SET1(6656.0f / 12825.0f);
const MFLOAT _28561_56430_ps = SET1(28561.0f / 56430.0f);
const MFLOAT _minus_9_50_ps = SET1(-9.0f / 50.0f);
const MFLOAT _2_55_ps = SET1(2.0f / 55.0f);

// differences c_hat - c
const MFLOAT diff_1_ps = SET1(1.0f / 150.0f);
const MFLOAT diff_3_ps = SET1(3.0f / 100.0f);
const MFLOAT diff_4_ps = SET1(-48.0f / 225.0f);
const MFLOAT diff_5_ps = SET1(-1.0f / 20.0f);
const MFLOAT diff_6_ps = SET1(6.0f / 25.0f);


class kerr_integrator {

	struct geodesic_data {
		MFLOAT r;
		MFLOAT p_r;
		MFLOAT theta;
		MFLOAT p_theta;
		MFLOAT phi;
	};

	// a
	MFLOAT spin_constant;

	// event horizon
	MFLOAT event_horizon;

	// step
	MFLOAT step_ps;

	// L, Q, E, k
	ALIGN std::vector<float> angular_momenta;
	ALIGN std::vector<float> carter_constants;
	ALIGN std::vector<float> total_energies;
	ALIGN std::vector<float> kappas;

	// r, phi, theta
	ALIGN std::vector<float> radii;
	ALIGN std::vector<float> phis;
	ALIGN std::vector<float> thetas;

	// p_r, p_theta
	ALIGN std::vector<float> p_r;
	ALIGN std::vector<float> p_theta;

	float get256_avx2(__m256 a, int idx) {
		__m128i vidx = _mm_cvtsi32_si128(idx); // vmovd
		__m256i vidx256 = _mm256_castsi128_si256(vidx); // no instructions
		__m256 shuffled = _mm256_permutevar8x32_ps(a, vidx256); // vpermps
		return _mm256_cvtss_f32(shuffled);
	}

	std::vector<float> _compute_kappa() {
		std::vector<float> kappas(total_energies.size());
		size_t N = total_energies.size();
		for (size_t i = 0; i < N; i+=subproblem_size) {
			MFLOAT carter_ps = LOAD(&carter_constants[i]);
			MFLOAT angular_momenta_ps = LOAD(&angular_momenta[i]);
	
			// Q+L^2
			MFLOAT q_l_squared_sum_ps = FMADD(angular_momenta_ps, angular_momenta_ps, carter_ps);

			MFLOAT total_energies_ps = LOAD(&total_energies[i]);
			MFLOAT total_energies_squared_minus_one_ps = FMSUB(total_energies_ps, total_energies_ps, one_ps);
			MFLOAT spin_squared_ps = MUL(spin_constant, spin_constant);

			// k = [Q + L^2] + [a^2(E^2-1)] 
			MFLOAT res_ps = FMADD(spin_squared_ps, total_energies_squared_minus_one_ps, q_l_squared_sum_ps);
			
			STORE(&kappas[i], res_ps);
		}

		return kappas;
	}

	// Note S_inv stands from 1 / sigma

	MFLOAT _compute_D(MFLOAT radius) {
		MFLOAT radius_minus_two_ps = SUB(radius, SET1(2.0f));
		MFLOAT spin_squared_ps = MUL(spin_constant, spin_constant);
		// r(r-2) + a^2
		return FMADD(radius, radius_minus_two_ps, spin_squared_ps);
	}

	MFLOAT _compute_S_inv(MFLOAT radius, MFLOAT theta) {
		MFLOAT radius_squared_ps = MUL(radius, radius);
		MFLOAT cos_theta_ps = COS(theta);
		MFLOAT spin_times_cos_ps = MUL(spin_constant, cos_theta_ps);
		// r^2 + (a cos(theta))*(a cos(theta))
		MFLOAT res = FMADD(spin_times_cos_ps, spin_times_cos_ps, radius_squared_ps);
		return RCP(res);
	}

	MFLOAT _compute_r_dot(MFLOAT D, MFLOAT S_inv, MFLOAT p_r) {
		return MUL(D, MUL(S_inv, p_r));
	}

	MFLOAT _compute_theta_dot(MFLOAT S_inv, MFLOAT p_theta) {
		return MUL(S_inv, p_theta);
	}

	MFLOAT _compute_phi_dot(MFLOAT radius, MFLOAT theta, MFLOAT D, MFLOAT E, MFLOAT L, MFLOAT S_inv) {
		MFLOAT S_ps = RCP(S_inv);
		// S-2r
		MFLOAT sigma_minus_2_r_ps = FMADD(minus_two_ps, radius, S_ps);
		MFLOAT sin_theta_ps = SIN(theta);
		MFLOAT sin_theta_squared_ps = MUL(sin_theta_ps, sin_theta_ps);
		// L / sin^2(theta)
		MFLOAT L_over_sin_theta_squared_ps =  MUL(L, RCP(sin_theta_squared_ps));
		// (S-2r) * (L / sin^2(theta))
		MFLOAT rhs_numerator_ps = MUL(sigma_minus_2_r_ps, L_over_sin_theta_squared_ps);
		MFLOAT two_a_ps = MUL(two_ps, spin_constant);
		MFLOAT r_times_energy_ps = MUL(radius, E);
		// 2arE + ((S-2r)L/sin^2(theta))
		MFLOAT numerator_ps = FMADD(two_a_ps, r_times_energy_ps, rhs_numerator_ps);
		// 1 / (S*D)
		MFLOAT D_inv_ps = RCP(D);
		MFLOAT denominator_ps = MUL(D_inv_ps, S_inv);

		return MUL(numerator_ps, denominator_ps);
	}

	MFLOAT _compute_p_r_dot(MFLOAT radius, MFLOAT D, MFLOAT S_inv, MFLOAT E, MFLOAT k, MFLOAT p_r, MFLOAT L) {
		MFLOAT radius_squared_ps = MUL(radius, radius);
		// r^2+a^2 - we'll reuse this later
		MFLOAT r_sqr_a_sqr_ps = FMADD(spin_constant, spin_constant, radius_squared_ps);
		MFLOAT two_times_r_sqr_a_sqr_ps = ADD(r_sqr_a_sqr_ps, r_sqr_a_sqr_ps);
		MFLOAT energy_squared_ps = MUL(E, E);

		// 2(r^2+a^2)E^2 - D
		MFLOAT second_term_first_bracket_ps = FMSUB(two_times_r_sqr_a_sqr_ps, energy_squared_ps, D);

		MFLOAT r_sqr_a_sqr_ps_kappa_ps = ADD(r_sqr_a_sqr_ps, k);
		MFLOAT r_minus_one_ps = SUB(radius, one_ps);

		// (r^2+a^2+k)(r-1)
		MFLOAT first_term_first_bracket_ps = MUL(r_sqr_a_sqr_ps_kappa_ps, r_minus_one_ps);

		// -(r^2+a^2+k)(r-1)+r(2(r^2+a^2)E^2 - D)
		MFLOAT sub_two_terms_first_bracket_ps = FMSUB(radius, second_term_first_bracket_ps, first_term_first_bracket_ps);

		// -2*a
		MFLOAT two_a_ps = MUL(minus_two_ps, spin_constant);
		MFLOAT energy_times_angular_momentum_ps = MUL(E, L);

		// -(r^2+a^2+k)(r-1)+r(2(r^2+a^2)E^2 - D) - 2aEL
		MFLOAT first_term_numerator_ps = FMADD(two_a_ps, energy_times_angular_momentum_ps, sub_two_terms_first_bracket_ps);

		MFLOAT p_r_squared_ps = MUL(p_r, p_r);
		MFLOAT two_p_r_squared_ps = MUL(two_ps, p_r_squared_ps);

		// 2(p_r)^2(r-1)
		MFLOAT second_term_second_bracket_ps = MUL(two_p_r_squared_ps, r_minus_one_ps);

		MFLOAT D_inv_ps = RCP(D);
		// [-(r^2+a^2+k)(r-1)+r(2(r^2+a^2)E^2 - D) - 2aEL]/D - 2p_r^2(r-1)
		MFLOAT total_numerator_ps = FMSUB(D_inv_ps, first_term_numerator_ps, second_term_second_bracket_ps);

		return MUL(S_inv, total_numerator_ps);
	}

	MFLOAT _compute_p_theta_dot(MFLOAT theta, MFLOAT S_inv, MFLOAT L, MFLOAT E) {
		MFLOAT sin_theta_ps = SIN(theta);

		MFLOAT energy_squared_minus_one_ps = FMSUB(E, E, one_ps);
		MFLOAT a_squared_ps = MUL(spin_constant, spin_constant);
		MFLOAT second_term_ps = MUL(a_squared_ps, energy_squared_minus_one_ps);

		MFLOAT sin_theta_squared_ps = MUL(sin_theta_ps, sin_theta_ps);
		MFLOAT sin_theta_reciprocal_squared_ps = RCP(sin_theta_squared_ps);
		MFLOAT first_term_ps = MUL(L, sin_theta_reciprocal_squared_ps);

		MFLOAT second_bracket_ps = FMSUB(first_term_ps, first_term_ps, second_term_ps);

		MFLOAT cos_theta_ps = COS(theta);
		MFLOAT res = MUL(cos_theta_ps, second_bracket_ps);
		res = MUL(res, sin_theta_ps);
		return MUL(res, S_inv);
		
	}

	geodesic_data _next_step_geodesic(size_t idx) {
		// Load data from last step (y_n in the RK4 notation)
		MFLOAT radius_ps = LOAD(&radii[idx]);
		MFLOAT D_ps = _compute_D(radius_ps);
		MFLOAT theta_ps = LOAD(&thetas[idx]);
		MFLOAT S_inv_ps = _compute_S_inv(radius_ps, theta_ps);
		MFLOAT p_r_ps = LOAD(&p_r[idx]);
		MFLOAT p_theta_ps = LOAD(&p_theta[idx]);
		MFLOAT phi_ps = LOAD(&phis[idx]);

		// Load static data - energy, kappa and angular momentum
		MFLOAT energy_ps = LOAD(&total_energies[idx]);
		MFLOAT angular_momentum_ps = LOAD(&angular_momenta[idx]);
		MFLOAT kappa_ps = LOAD(&kappas[idx]);

		// Note that all k_i must be multiplied through by h
		
		//K1 -------------------------------------------------------------------
		MFLOAT radius_k1_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, p_r_ps));
		MFLOAT phi_k1_ps = MUL(step_ps, _compute_phi_dot(radius_ps, theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k1_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, p_theta_ps));
		MFLOAT p_r_k1_ps = MUL(step_ps, _compute_p_r_dot(radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k1_ps = MUL(step_ps, _compute_p_theta_dot(theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k1 = { radius_k1_ps, phi_k1_ps, theta_k1_ps, p_r_k1_ps, p_theta_k1_ps };
		
		//K2 --------------------------------------------------------------------
		MFLOAT adjusted_radius_ps = FMADD(_1_4_ps, radius_k1_ps, radius_ps);
		MFLOAT adjusted_phi_ps = FMADD(_1_4_ps, phi_k1_ps, phi_ps);
		MFLOAT adjusted_theta_ps = FMADD(_1_4_ps, theta_k1_ps, theta_ps);
		MFLOAT adjusted_p_r_ps = FMADD(_1_4_ps, p_r_k1_ps, p_r_ps);
		MFLOAT adjusted_p_theta_ps = FMADD(_1_4_ps, p_theta_k1_ps, p_theta_ps);

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k2_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps));
		MFLOAT phi_k2_ps = MUL(step_ps, _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k2_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, adjusted_p_theta_ps));
		MFLOAT p_r_k2_ps = MUL(step_ps, _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k2_ps = MUL(step_ps, _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k2 = { radius_k2_ps, phi_k2_ps, theta_k2_ps, p_r_k2_ps, p_theta_k2_ps };

		//K3 --------------------------------------------------------------------
		adjusted_radius_ps = FMADD(_9_32_ps, radius_k2_ps, FMADD(_3_32_ps, radius_k1_ps, radius_ps));
		adjusted_phi_ps = FMADD(_9_32_ps, phi_k2_ps, FMADD(_3_32_ps, phi_k1_ps, phi_ps));
		adjusted_theta_ps = FMADD(_9_32_ps, theta_k2_ps, FMADD(_3_32_ps, theta_k1_ps, theta_ps));
		adjusted_p_r_ps = FMADD(_9_32_ps, p_r_k2_ps, FMADD(_3_32_ps, p_r_k1_ps, p_r_ps));
		adjusted_p_theta_ps = FMADD(_9_32_ps, p_theta_k2_ps, FMADD(_3_32_ps, p_theta_k1_ps, p_theta_ps));

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k3_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps));
		MFLOAT phi_k3_ps = MUL(step_ps, _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k3_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, adjusted_p_theta_ps));
		MFLOAT p_r_k3_ps = MUL(step_ps, _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k3_ps = MUL(step_ps, _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k3 = { radius_k3_ps, phi_k3_ps, theta_k3_ps, p_r_k3_ps, p_theta_k3_ps };

		//K4 ---------------------------------------------------------------------
		adjusted_radius_ps = FMADD(_7296_2197_ps, radius_k3_ps, FMADD(_minus_7200_2197_ps, radius_k2_ps, FMADD(_1932_2197_ps, radius_k1_ps, radius_ps)));
		adjusted_phi_ps = FMADD(_7296_2197_ps, phi_k3_ps, FMADD(_minus_7200_2197_ps, phi_k2_ps, FMADD(_1932_2197_ps, phi_k1_ps, phi_ps)));
		adjusted_theta_ps = FMADD(_7296_2197_ps, theta_k3_ps, FMADD(_minus_7200_2197_ps, theta_k2_ps, FMADD(_1932_2197_ps, theta_k1_ps, theta_ps)));
		adjusted_p_r_ps = FMADD(_7296_2197_ps, p_r_k3_ps, FMADD(_minus_7200_2197_ps, p_r_k2_ps, FMADD(_1932_2197_ps, p_r_k1_ps, p_r_ps)));
		adjusted_p_theta_ps = FMADD(_7296_2197_ps, p_theta_k3_ps, FMADD(_minus_7200_2197_ps, p_theta_k2_ps, FMADD(_1932_2197_ps, p_theta_k1_ps, p_theta_ps)));

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k4_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps));
		MFLOAT phi_k4_ps = MUL(step_ps, _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k4_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, adjusted_p_theta_ps));
		MFLOAT p_r_k4_ps = MUL(step_ps, _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k4_ps = MUL(step_ps, _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k4 = { radius_k4_ps, phi_k4_ps, theta_k4_ps, p_r_k4_ps, p_theta_k4_ps };

		// K5 ----------------------------------------------------------------------
		adjusted_radius_ps = FMADD(_minus_845_4104_ps, radius_k4_ps, FMADD(_3680_513_ps, radius_k3_ps, FMADD(_minus_8_ps, radius_k2_ps, FMADD(_439_216_ps, radius_k1_ps, radius_ps))));
		adjusted_phi_ps = FMADD(_minus_845_4104_ps, phi_k4_ps, FMADD(_3680_513_ps, phi_k3_ps, FMADD(_minus_8_ps, phi_k2_ps, FMADD(_439_216_ps, phi_k1_ps, phi_ps))));
		adjusted_theta_ps = FMADD(_minus_845_4104_ps, theta_k4_ps, FMADD(_3680_513_ps, theta_k3_ps, FMADD(_minus_8_ps, theta_k2_ps, FMADD(_439_216_ps, theta_k1_ps, theta_ps))));
		adjusted_p_r_ps = FMADD(_minus_845_4104_ps, p_r_k4_ps, FMADD(_3680_513_ps, p_r_k3_ps, FMADD(_minus_8_ps, p_r_k2_ps, FMADD(_439_216_ps, p_r_k1_ps, p_r_ps))));
		adjusted_p_theta_ps = FMADD(_minus_845_4104_ps, p_theta_k4_ps, FMADD(_3680_513_ps, p_theta_k3_ps, FMADD(_minus_8_ps, p_theta_k2_ps, FMADD(_439_216_ps, p_theta_k1_ps, p_theta_ps))));

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k5_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps));
		MFLOAT phi_k5_ps = MUL(step_ps, _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k5_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, adjusted_p_theta_ps));
		MFLOAT p_r_k5_ps = MUL(step_ps, _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k5_ps = MUL(step_ps, _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k5 = { radius_k5_ps, phi_k5_ps, theta_k5_ps, p_r_k5_ps, p_theta_k5_ps };

		// K6 -----------------------------------------------------------------------
		adjusted_radius_ps = FMADD(_minus_11_40_ps, radius_k5_ps, FMADD(_1859_4104_ps, radius_k4_ps, FMADD(_minus_3544_2565_ps, radius_k3_ps, FMADD(two_ps, radius_k2_ps, FMADD(_minus_8_27_ps, radius_k1_ps, radius_ps)))));
		adjusted_phi_ps = FMADD(_minus_11_40_ps, phi_k5_ps, FMADD(_1859_4104_ps, phi_k4_ps, FMADD(_minus_3544_2565_ps, phi_k3_ps, FMADD(two_ps, phi_k2_ps, FMADD(_minus_8_27_ps, phi_k1_ps, phi_ps)))));
		adjusted_theta_ps = FMADD(_minus_11_40_ps, theta_k5_ps, FMADD(_1859_4104_ps, theta_k4_ps, FMADD(_minus_3544_2565_ps, theta_k3_ps, FMADD(two_ps, theta_k2_ps, FMADD(_minus_8_27_ps, theta_k1_ps, theta_ps)))));
		adjusted_p_r_ps = FMADD(_minus_11_40_ps, p_r_k5_ps, FMADD(_1859_4104_ps, p_r_k4_ps, FMADD(_minus_3544_2565_ps, p_r_k3_ps, FMADD(two_ps, p_r_k2_ps, FMADD(_minus_8_27_ps, p_r_k1_ps, p_r_ps)))));
		adjusted_p_theta_ps = FMADD(_minus_11_40_ps, p_theta_k5_ps, FMADD(_1859_4104_ps, p_theta_k4_ps, FMADD(_minus_3544_2565_ps, p_theta_k3_ps, FMADD(two_ps, p_theta_k2_ps, FMADD(_minus_8_27_ps, p_theta_k1_ps, p_theta_ps)))));

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k6_ps = MUL(step_ps, _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps));
		MFLOAT phi_k6_ps = MUL(step_ps, _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps));
		MFLOAT theta_k6_ps = MUL(step_ps, _compute_theta_dot(S_inv_ps, adjusted_p_theta_ps));
		MFLOAT p_r_k6_ps = MUL(step_ps, _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps));
		MFLOAT p_theta_k6_ps = MUL(step_ps, _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps));

		std::array<MFLOAT, 5> k6 = { radius_k6_ps, phi_k6_ps, theta_k6_ps, p_r_k6_ps, p_theta_k6_ps };

		// compute the difference of as a vector and compute the L2 norm - this will be our error
		auto diff = [&k1, &k3, &k4, &k5, &k6](size_t i) {
			//sum_i = 1_to_6(c_hat(i) - c(i)) * k_i
			return FMADD(k6[i], diff_6_ps, FMADD(k5[i], diff_5_ps, FMADD(k4[i], diff_4_ps, FMADD(k3[i], diff_3_ps, MUL(k1[i],diff_1_ps)))));
		};

		std::array<MFLOAT, 5> diff_k = create_array<MFLOAT, 5>(diff);

		if (get256_avx2(radius_ps, 7) < 1.97f)
			std::cout << "STOP";

		MFLOAT error_ps = _compute_L2_norm<5>(diff_k);

		// compare radius with event horizon with a mask - it's given by 1+sqrt(1-a^2)
		MFLOAT event_horizon_radius_dist_ps = _compute_L2_norm<1>(std::array<MFLOAT, 1> { SUB(radius_ps, event_horizon) });
		MFLOAT event_horizon_mask = CMP(event_horizon_radius_dist_ps, tolerance_ps, _CMP_LT_OQ);

		// blend error_ps with tolerance_ps based on that mask
		error_ps = BLEND(error_ps, tolerance_ps, event_horizon_mask);

		// compare with tolerance
		MFLOAT above_error_mask = CMP(error_ps, tolerance_ps, _CMP_GT_OQ);

		// new step size:
		// new_step = 0.9 * previous_step * fourth_root(tolerance / error)
		MFLOAT tolerance_over_error_ps = MUL(tolerance_ps, RCP(error_ps));
		MFLOAT fourth_root_ps = SQRT(SQRT(tolerance_over_error_ps));
		step_ps = MUL(MUL(zero_point_nine_ps, step_ps), fourth_root_ps);

		// clamp step_ps
		step_ps = MIN(step_ps, SET1(1.5f));
		step_ps = MAX(step_ps, SET1(0.0001f));

		// set step_ps to zero for those
		step_ps = BLEND(step_ps, SETZERO, event_horizon_mask);

		// should we recompute
		bool recompute = !TESTZ(above_error_mask, above_error_mask);

		if (recompute) {
			auto [_radius_ps, _p_r_ps, _theta_ps, _p_theta_ps, _phi_ps] = _next_step_geodesic(idx);
			radius_ps = _radius_ps;
			p_r_ps = _p_r_ps;
			theta_ps = _theta_ps;
			p_theta_ps = _p_theta_ps;
			phi_ps = _phi_ps;
		} else {
			radius_ps = FMADD(_minus_1_5_ps, radius_k5_ps, FMADD(_2197_4104_ps, radius_k4_ps, FMADD(_1408_2565_ps, radius_k3_ps, FMADD(_25_216_ps, radius_k1_ps, radius_ps))));
			phi_ps = FMADD(_minus_1_5_ps, phi_k5_ps, FMADD(_2197_4104_ps, phi_k4_ps, FMADD(_1408_2565_ps, phi_k3_ps, FMADD(_25_216_ps, phi_k1_ps, phi_ps))));
			theta_ps = FMADD(_minus_1_5_ps, theta_k5_ps, FMADD(_2197_4104_ps, theta_k4_ps, FMADD(_1408_2565_ps, theta_k3_ps, FMADD(_25_216_ps, theta_k1_ps, theta_ps))));
			p_r_ps = FMADD(_minus_1_5_ps, p_r_k5_ps, FMADD(_2197_4104_ps, p_r_k4_ps, FMADD(_1408_2565_ps, p_r_k3_ps, FMADD(_25_216_ps, p_r_k1_ps, p_r_ps))));
			p_theta_ps = FMADD(_minus_1_5_ps, p_theta_k5_ps, FMADD(_2197_4104_ps, p_theta_k4_ps, FMADD(_1408_2565_ps, p_theta_k3_ps, FMADD(_25_216_ps, p_theta_k1_ps, p_theta_ps))));
		}

		return {radius_ps, p_r_ps, theta_ps, p_theta_ps, phi_ps};
	}

public:

	kerr_integrator(float									  a,
					initial_particle_data<spacetime::kerr> _initial_data)
		: spin_constant(SET1(a))
		, event_horizon(SET1(1.0f+std::sqrtf(1.0f-a*a)))
		, step_ps(SET1(1.0f))
		, angular_momenta(std::move(_initial_data.angular_momenta))
		, carter_constants(std::move(_initial_data.carter_constants))
		, total_energies(std::move(_initial_data.energies))
		, kappas(_compute_kappa())
		, radii(std::move(_initial_data.initial_radii))
		, phis(std::move(_initial_data.initial_phis))
		, thetas(std::move(_initial_data.initial_thetas))
		, p_r(std::move(_initial_data.initial_p_r))
		, p_theta(std::move(_initial_data.initial_p_theta)) {

		size_t N = radii.size();
		assert(angular_momenta.size() == N, "Angular momenta must have the same size as radii");
		assert(phis.size() == N, "Phi values must have the same size as radii");
		assert(carter_constants.size() == N, "Carter constants must have the same size as radii");
		assert(total_energies.size() == N, "Total energies must have the same size as radii");
		assert(thetas.size() == N, "Theta values must have the same size as radii");
		assert(p_r.size() == N, "Radial momenta must have the same size as radii");
		assert(p_theta.size() == N, "Momenta along the theta direction must have the same size as radii");
		assert(a >= 0.0f && a <= 1.0f, "Constant a must be between 0 and 1");
	}

	geodesic_data next_geodesic(size_t idx) {
		auto [r, pr, th, p_th, phi] = _next_step_geodesic(idx);

		STORE(&radii[idx], r);
		STORE(&p_r[idx], pr);
		STORE(&thetas[idx], th);
		STORE(&p_theta[idx], p_th);
		STORE(&phis[idx], phi);

		return { r, pr, th, p_th, phi};
	}

	void send_data() {
		message_kerr data(radii.size());
		size_t N = radii.size();
		for (size_t i = 0; i < N; i += subproblem_size) {
			auto [r, p_r, th, p_th, phi] = next_geodesic(i);
			data.convert_and_add(r, phi, th, spin_constant, i);
		}

		data.send();
	}

	void send_initial_data() {
		message_kerr data(radii.size());
		size_t N = radii.size();
		for (size_t i = 0; i < N; i += subproblem_size) {
			auto r = LOAD(&radii[i]);
			auto th = LOAD(&thetas[i]);
			auto phi = LOAD(&phis[i]);
			data.convert_and_add(r, phi, th, spin_constant, i);
		}

		data.send();
	}

	void rock_n_roll() {
		send_initial_data();
		for (;;)
			send_data();
	}

	void print_radii() {
		std::cout << "Radii: ";
		size_t N = radii.size();
		for (int i = 0; i < N; ++i)
			std::cout << radii[i] << " ";

		std::cout << std::endl;
	}

};
