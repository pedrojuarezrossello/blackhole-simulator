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
constexpr float step = 5.0f; 

template<size_t N>
class integrator {
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
		integrator(float                _black_hole_mass,
				   initial_particle_data<N> _initial_data)
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
			message<N> data = {};
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
			message<N> data = {};
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
