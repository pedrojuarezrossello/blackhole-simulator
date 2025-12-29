#include "ofMain.h"
#include "ofApp.h"
#include "utils.h"
#include "integrator.h"
#include "message.h"
#include "message_queue.h"
#include <thread>

message_queue<message_kerr<N>> data_queue;

// Angular momentum must be >= sqrt(12)*M

int main( ){
	// Set up integrator
	initial_particle_data<spacetime::kerr, N> particle_data;

	particle_data.initial_radii = create_array<float, N>([](size_t i) { return 7.0f+i*0.1f; });
	particle_data.initial_p_r = create_array<float, N>([](size_t i) { return 0.0f; });
	particle_data.initial_phis = create_array<float, N>([](size_t i) { return 0.0f; });
	particle_data.initial_thetas = create_array<float, N>([](size_t i) {return 3.141592f / 4.0f;});
	particle_data.initial_p_theta = create_array<float, N>([] (size_t i) { return 1.9558f; });
	particle_data.angular_momenta = create_array<float, N>([](size_t i) { return 2.37176f; });
	particle_data.carter_constants = create_array<float, N>([](size_t i) { return 3.82514f; });
	particle_data.energies = create_array<float, N>([](size_t i) { return 0.935179f; });

	constexpr float spin = 0.9f;
	kerr_integrator<N> solver(spin, particle_data);
	 
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
