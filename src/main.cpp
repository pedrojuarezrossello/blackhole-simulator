#include "ofMain.h"
#include "ofApp.h"
#include "utils.h"
#include "integrator.h"
#include "message.h"
#include "message_queue.h"
#include <thread>

message_queue<message_kerr> data_queue;

// todo - extend to non multiples of 8
// todo - handle event horizon

int main(int argc, char * argv[]) {
	// Set up integrator
	initial_particle_data<spacetime::kerr> particle_data("C:\\Users\\Pedro\\Downloads\\of_v0.12.1_vs_64_release\\apps\\myApps\\schwarzschild_black_hole\\src\\data.txt");

	const float spin = argc > 1 ? atof(argv[1]) : get_default(particle_data);
	kerr_integrator solver(spin, particle_data);
	 
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
