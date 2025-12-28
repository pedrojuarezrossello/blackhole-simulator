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

// This will be adjusted later
constexpr float step = 0.1f;


template<size_t N>
class schwarzschild_integrator {
	// It's negative!!
	MFLOAT _2black_hole_mass;

	ALIGN std::array<float, N> angular_momenta;
	ALIGN std::array<float, N> radii;
	ALIGN std::array<float, N> phis;
	ALIGN std::array<float, N> total_energies;
	ALIGN std::array<float, N> directions;

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
	
		//Update the corresponding radii to wiggle them away from the energy peak where E^2-V^2<=0
		while (!_mm256_testz_ps(nonnegative_mask, nonnegative_mask)) {
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
		// a^2 - b^2 = (a-b)*(a+b)
		MFLOAT potential = _potential_energy(arg_for_potential, idx);
		return MUL(SUB(total_energy, potential), ADD(total_energy, potential));
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
		// Compute radius inverse with a built-in intrisic - note it's an approximation...
		MFLOAT radius_inv_ps = RCP(radius);
		MFLOAT radius_inv_squared_ps = MUL(radius_inv_ps, radius_inv_ps);
		// Note that _2black_hole_mass is negative
		MFLOAT first_term_ps = FMADD(_2black_hole_mass, radius_inv_ps, one_ps);
		MFLOAT angular_momenta_ps = LOAD(&angular_momenta[idx]);
		MFLOAT second_term_ps = FMADD(MUL(angular_momenta_ps, angular_momenta_ps), radius_inv_squared_ps, one_ps);
		return MUL(first_term_ps, second_term_ps);
	}
	std::array<float, N> _initialise_energies(const std::array<float, N>& radii) {
		alignas(64) std::array<float, N> energies = {};
		// We loop through the array in chunks of subproblem_size<N>
		for (size_t i = 0; i < N; i += subproblem_size<N>) {
			// Load the initial radii in the ZMM register
			MFLOAT initial_radii_ps = LOAD(&radii[i]);
			// Compute the potential energies for this chunk
			MFLOAT potential_energies_ps = _potential_energy(initial_radii_ps, i);
			// Store the total energy
			STORE(&energies[i], potential_energies_ps);
		}
		return energies;
	}

	public:
		schwarzschild_integrator(float                _black_hole_mass,
			initial_particle_data<spacetime::schwarzschild, N> _initial_data)
			: _2black_hole_mass(SET1(-2 * _black_hole_mass))
			, angular_momenta(std::move(_initial_data.angular_momenta))
			, radii(std::move(_initial_data.initial_radii))
			, phis(std::move(_initial_data.initial_phis))
			, total_energies(_initialise_energies(radii))
			, directions(create_array<float, N>([](size_t i) {return -1.0f;}))
		{ }

		// We'll probs have to use an adaptive step scheme...
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
			std::cout << "Radii: ";
			for (size_t i = 0; i < N; ++i)
				std::cout << radii[i] << " ";
			std::cout << std::endl;
		}

		void send_data() {
			message_schwarzschild<N> data = {};
			for (size_t i = 0; i < N; i += subproblem_size<N>) {
				MFLOAT radii_chunk = next_radius(step, i);
				_update_directions(&radii_chunk, i);
				MFLOAT phis_chunk = next_phi(step, i);
				MFLOAT thetas_chunk = SET1(3.141592f / 2.0f); 
				data.convert_and_add(radii_chunk, phis_chunk, thetas_chunk, _2black_hole_mass, i);
			}

			data.send();
		}

		void send_initial_data() {
			message_schwarzschild<N> data = {};
			for (size_t i = 0; i < N; i += subproblem_size<N>) {
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
*  Here p stands for the linear momentum as usual. We'll solve them using RK4 as before.
* 
*/

const MFLOAT two_ps = SET1(2.0f);
const MFLOAT minus_two_ps = SET1(-2.0f);
const MFLOAT one_ps = SET1(1.0f);

template <size_t N>
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

	// L, Q, E, k
	ALIGN std::array<float, N> angular_momenta;
	ALIGN std::array<float, N> carter_constants;
	ALIGN std::array<float, N> total_energies;
	ALIGN std::array<float, N> kappas;

	// r, phi, theta
	ALIGN std::array<float, N> radii;
	ALIGN std::array<float, N> phis;
	ALIGN std::array<float, N> thetas;

	// p_r, p_theta
	ALIGN std::array<float, N> p_r;
	ALIGN std::array<float, N> p_theta;

	std::array<float, N> _compute_kappa() {
		std::array<float, N> kappas = {};

		for (size_t i = 0; i < N; i+=subproblem_size<N>) {
			MFLOAT carter_ps = LOAD(&carter_constants[i]);
			MFLOAT angular_momenta_ps = LOAD(&angular_momenta[i]);
	
			// Q+L^2
			MFLOAT q_l_squared_sum_ps = FMADD(angular_momenta_ps, angular_momenta_ps, carter_ps);

			MFLOAT total_energies_ps = LOAD(&total_energies[i]);
			MFLOAT total_energies_squared_minus_one_ps = FMSUB(total_energies_ps, total_energies_ps, one_ps);
			MFLOAT spin_squared_ps = MUL(spin_constant, spin_constant);

			// k = [Q + L^2] + [a^2(E^2-1)] 
			MFLOAT res_ps = FMADD(spin_squared_ps, total_energies_squared_minus_one_ps, q_l_squared_sum_ps);
			STREAM(&kappas[i], res_ps);
		}

		return kappas;
	}

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

	geodesic_data _next_step_geodesic(float step, size_t idx) {
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

		//K1 -------------------------------------------------------------------
		MFLOAT radius_k1_ps = _compute_r_dot(D_ps, S_inv_ps, p_r_ps);
		MFLOAT phi_k1_ps = _compute_phi_dot(radius_ps, theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps);
		MFLOAT theta_k1_ps = _compute_theta_dot(S_inv_ps, p_theta_ps);
		MFLOAT p_r_k1_ps = _compute_p_r_dot(radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, p_r_ps, angular_momentum_ps);
		MFLOAT p_theta_k1_ps = _compute_p_theta_dot(theta_ps, S_inv_ps, angular_momentum_ps, energy_ps);

		// Load steps
		MFLOAT step_ps = SET1(step);
		MFLOAT half_step_ps = SET1(step / 2.0f);

		//K2 --------------------------------------------------------------------
		MFLOAT adjusted_radius_ps = FMADD(half_step_ps, radius_k1_ps, radius_ps);
		MFLOAT adjusted_phi_ps = FMADD(half_step_ps, phi_k1_ps, phi_ps);
		MFLOAT adjusted_theta_ps = FMADD(half_step_ps, theta_k1_ps, theta_ps);
		MFLOAT adjusted_p_r_ps = FMADD(half_step_ps, p_r_k1_ps, p_r_ps);
		MFLOAT adjusted_p_theta_ps = FMADD(half_step_ps, p_theta_k1_ps, p_theta_ps);

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k2_ps = _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps);
		MFLOAT phi_k2_ps = _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps);
		MFLOAT theta_k2_ps = _compute_theta_dot(S_inv_ps, adjusted_theta_ps);
		MFLOAT p_r_k2_ps = _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps);
		MFLOAT p_theta_k2_ps = _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps);

		//K3 --------------------------------------------------------------------
		adjusted_radius_ps = FMADD(half_step_ps, radius_k2_ps, radius_ps);
		adjusted_phi_ps = FMADD(half_step_ps, phi_k2_ps, phi_ps);
		adjusted_theta_ps = FMADD(half_step_ps, theta_k2_ps, theta_ps);
		adjusted_p_r_ps = FMADD(half_step_ps, p_r_k2_ps, p_r_ps);
		adjusted_p_theta_ps = FMADD(half_step_ps, p_theta_k2_ps, p_theta_ps);

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k3_ps = _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps);
		MFLOAT phi_k3_ps = _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps);
		MFLOAT theta_k3_ps = _compute_theta_dot(S_inv_ps, adjusted_theta_ps);
		MFLOAT p_r_k3_ps = _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps);
		MFLOAT p_theta_k3_ps = _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps);

		//K4 ---------------------------------------------------------------------
		adjusted_radius_ps = FMADD(step_ps, radius_k3_ps, radius_ps);
		adjusted_phi_ps = FMADD(step_ps, phi_k3_ps, phi_ps);
		adjusted_theta_ps = FMADD(step_ps, theta_k3_ps, theta_ps);
		adjusted_p_r_ps = FMADD(step_ps, p_r_k3_ps, p_r_ps);
		adjusted_p_theta_ps = FMADD(step_ps, p_theta_k3_ps, p_theta_ps);

		// Recalculate S and D
		D_ps = _compute_D(adjusted_radius_ps);
		S_inv_ps = _compute_S_inv(adjusted_radius_ps, adjusted_phi_ps);
		MFLOAT radius_k4_ps = _compute_r_dot(D_ps, S_inv_ps, adjusted_p_r_ps);
		MFLOAT phi_k4_ps = _compute_phi_dot(adjusted_radius_ps, adjusted_theta_ps, D_ps, energy_ps, angular_momentum_ps, S_inv_ps);
		MFLOAT theta_k4_ps = _compute_theta_dot(S_inv_ps, adjusted_theta_ps);
		MFLOAT p_r_k4_ps = _compute_p_r_dot(adjusted_radius_ps, D_ps, S_inv_ps, energy_ps, kappa_ps, adjusted_p_r_ps, angular_momentum_ps);
		MFLOAT p_theta_k4_ps = _compute_p_theta_dot(adjusted_theta_ps, S_inv_ps, angular_momentum_ps, energy_ps);

		// Sum k's
		MFLOAT radius_k1_4_ps = ADD(radius_k1_ps, radius_k4_ps);
		MFLOAT phi_k1_4_ps = ADD(phi_k1_ps, phi_k4_ps);
		MFLOAT theta_k1_4_ps = ADD(theta_k1_ps, theta_k4_ps);
		MFLOAT p_r_k1_4_ps = ADD(p_r_k1_ps, p_r_k4_ps);
		MFLOAT p_theta_k1_4_ps = ADD(p_theta_k1_ps, p_theta_k4_ps);

		MFLOAT radius_k2_3_ps = ADD(radius_k2_ps, radius_k3_ps);
		MFLOAT phi_k2_3_ps = ADD(phi_k2_ps, phi_k3_ps);
		MFLOAT theta_k2_3_ps = ADD(theta_k2_ps, theta_k3_ps);
		MFLOAT p_r_k2_3_ps = ADD(p_r_k2_ps, p_r_k3_ps);
		MFLOAT p_theta_k2_3_ps = ADD(p_theta_k2_ps, p_theta_k3_ps);

		MFLOAT radius_sum_k_ps = FMADD(two_ps, radius_k2_3_ps, radius_k1_4_ps);
		MFLOAT phi_sum_k_ps = FMADD(two_ps, phi_k2_3_ps, phi_k1_4_ps);
		MFLOAT theta_sum_k_ps = FMADD(two_ps, theta_k2_3_ps, theta_k1_4_ps);
		MFLOAT p_r_sum_k_ps = FMADD(two_ps, p_r_k2_3_ps, p_r_k1_4_ps);
		MFLOAT p_theta_sum_k_ps = FMADD(two_ps, p_theta_k2_3_ps, p_theta_k1_4_ps);

		MFLOAT sixth_step_ps = SET1(step/6.0f);

		radius_ps = FMADD(sixth_step_ps, radius_sum_k_ps, radius_ps);
		phi_ps = FMADD(sixth_step_ps, phi_sum_k_ps, phi_ps);
		theta_ps = FMADD(sixth_step_ps, theta_sum_k_ps, theta_ps);
		p_r_ps = FMADD(sixth_step_ps, p_r_sum_k_ps, p_r_ps);
		p_theta_ps = FMADD(sixth_step_ps, p_theta_sum_k_ps, p_theta_ps);

		return {radius_ps, p_r_ps, theta_ps, p_theta_ps, phi_ps};
	}

public:

	kerr_integrator(float									  a,
					initial_particle_data<spacetime::kerr, N> _initial_data)
		: spin_constant(SET1(a))
		, angular_momenta(std::move(_initial_data.angular_momenta))
		, carter_constants(std::move(_initial_data.carter_constants))
		, total_energies(std::move(_initial_data.energies))
		, kappas(_compute_kappa())
		, radii(std::move(_initial_data.initial_radii))
		, phis(std::move(_initial_data.initial_phis))
		, thetas(std::move(_initial_data.initial_thetas))
		, p_r(std::move(_initial_data.initial_p_r))
		, p_theta(std::move(_initial_data.initial_p_theta)) {}


	geodesic_data next_geodesic(float step, size_t idx) {
		// calc step here 
		auto [r, pr, th, p_th, phi] = _next_step_geodesic(step, idx);

		STORE(&radii[idx], r);
		STORE(&p_r[idx], pr);
		STORE(&thetas[idx], th);
		STORE(&p_theta[idx], p_th);
		STORE(&phis[idx], phi);

		return { r, pr, th, p_th, phi };
	}

	void send_data() {
	//	print_radii();
		message_kerr<N> data = {};
		for (size_t i = 0; i < N; i += subproblem_size<N>) {
			auto [r, p_r, th, p_th, phi] = next_geodesic(step, i);
			data.convert_and_add(r, phi, th, spin_constant, i);
		}

		data.send();
	}

	void send_initial_data() {
		message_kerr<N> data = {};
		for (size_t i = 0; i < N; i += subproblem_size<N>) {
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
		for (size_t i = 0; i < N; ++i)
			std::cout << radii[i] << " ";
		std::cout << std::endl;
	}

};
