#include "ofMain.h"
#include "ofApp.h"
#include "utils.h"
#include "integrator.h"
#include "message.h"
#include "message_queue.h"
#include <thread>

message_queue<message> data_queue;

int main(int argc, char * argv[]) {
	// Set up integrator
	initial_particle_data<schwarzschild> particle_data("data.txt");

	const float spin = argc > 1 ? atof(argv[1]) : get_default(particle_data);
	schwarzschild_integrator solver(spin, particle_data);
	
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
