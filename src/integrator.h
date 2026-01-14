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
	float bm_scalar;
	ALIGN std::vector<float> angular_momenta;
	ALIGN std::vector<float> radii;
	ALIGN std::vector<float> phis;
	ALIGN std::vector<float> total_energies;
	ALIGN std::vector<float> directions;

	void _update_directions(MFLOAT * new_radii_ps, size_t idx);

	MFLOAT _rhs_radius_ode_squared(MFLOAT total_energy, MFLOAT arg_for_potential, size_t idx);

	MFLOAT _next_step_radius(MFLOAT prev_radius, float step, size_t idx);

	MFLOAT _next_step_phi(MFLOAT radius, MFLOAT prev_phi, float step, size_t idx);

	MFLOAT _potential_energy(MFLOAT radius, size_t idx);

	std::vector<float> _initialise_energies(const std::vector<float> & radii);
	
	float _potential_energy_scalar(float radius, size_t idx);

	float _rhs_ode_squared_scalar(float arg_for_potential, size_t idx);

	float _next_step_radius_scalar(float step, size_t idx);

	void _update_directions_scalar(size_t idx);

	float _next_step_phi_scalar(float step, size_t idx);

	public:
		schwarzschild_integrator(float _black_hole_mass, initial_particle_data<schwarzschild> _initial_data);

		MFLOAT next_radius(float step, size_t idx);

		MFLOAT next_phi(float step, size_t idx);

		void send_data();

		void send_initial_data();

		void rock_n_roll();
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

const float tolerance = 0.05f;
const MFLOAT tolerance_ps = SET1(tolerance);

class kerr_integrator {

	struct geodesic_data {
		MFLOAT r;
		MFLOAT p_r;
		MFLOAT theta;
		MFLOAT p_theta;
		MFLOAT phi;
	};

	struct geodesic_data_scalar {
		float p;
		float p_r;
		float theta;
		float p_theta;
		float phi;
	};

	// a
	MFLOAT spin_constant;
	float a;

	// event horizon
	MFLOAT event_horizon;

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

	// step
	ALIGN std::vector<float> step;

	std::vector<float> _compute_kappa();

	// Note S_inv stands from 1 / sigma

	MFLOAT _compute_D(MFLOAT radius);

	MFLOAT _compute_S_inv(MFLOAT radius, MFLOAT cos_theta_ps);

	MFLOAT _compute_r_dot(MFLOAT D, MFLOAT S_inv, MFLOAT p_r);

	MFLOAT _compute_theta_dot(MFLOAT S_inv, MFLOAT p_theta);

	MFLOAT _compute_phi_dot(MFLOAT radius, MFLOAT sin_theta_ps, MFLOAT D, MFLOAT E, MFLOAT L, MFLOAT S_inv);

	MFLOAT _compute_p_r_dot(MFLOAT radius, MFLOAT D, MFLOAT S_inv, MFLOAT E, MFLOAT k, MFLOAT p_r, MFLOAT L);

	MFLOAT _compute_p_theta_dot(MFLOAT sin_theta_ps, MFLOAT cos_theta_ps, MFLOAT S_inv, MFLOAT L, MFLOAT E);

	geodesic_data _next_step_geodesic(size_t idx);

	// Imitate the do while
	geodesic_data_scalar _next_step_geodesic_scalar(size_t idx);
	
 public:
	kerr_integrator(float _a,
		initial_particle_data<kerr> _initial_data);

	geodesic_data next_geodesic(size_t idx);

	void send_data();

	void send_initial_data();

	void rock_n_roll();
};
