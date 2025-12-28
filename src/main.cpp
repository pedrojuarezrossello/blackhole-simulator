#include "ofMain.h"
#include "ofApp.h"
#include "utils.h"
#include "integrator.h"
#include "message.h"
#include "message_queue.h"
#include <thread>

message_queue<message<N>> data_queue;

// Angular momentum must be >= sqrt(12)*M

int main( ){
	// Set up integrator
	initial_particle_data<N> particle_data = {
		.initial_radii = create_array<float, N>([](size_t i) { return 6.0f + i * 2.0f; }),
		.initial_phis = create_array<float, N>([](size_t i) { return 0.0f; }),
		.initial_thetas = create_array<float, N>([](size_t i) { return 3.141592f / 2.0f; }),
		.angular_momenta = create_array<float, N>([](size_t i) { return std::sqrtf(13.0f) + i * 0.3f; })
	};

	 constexpr float black_hole_mass = 1.0f;
	integrator<N> solver(black_hole_mass, particle_data);
	 
	// Set up openFrameworks window
	ofGLWindowSettings settings;
	settings.setSize(2400, 1200);
	settings.windowMode = OF_WINDOW; 

	auto window = ofCreateWindow(settings);
	ofRunApp(window, std::make_shared<ofApp>());

	// Let's go
	 auto integrator_thread = std::jthread([&solver]() {
		solver.rock_n_roll();
	});
	ofRunMainLoop();
}
