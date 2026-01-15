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
	initial_particle_data<kerr> particle_data("data.txt");

	const float param = argc > 1 ? atof(argv[1]) : get_default(particle_data);
	kerr_integrator solver(param, particle_data);

	// Set up openFrameworks window
	ofGLWindowSettings settings;
	settings.setSize(2400, 1200);
	settings.windowMode = OF_WINDOW;

	auto window = ofCreateWindow(settings);

	size_t N = particle_data.initial_radii.size();

	using spacetime = std::conditional_t<std::is_same_v<decltype(particle_data), initial_particle_data<kerr>>, kerr, schwarzschild>;

	ofRunApp(window, std::make_shared<ofApp>(N, param, spacetime{}));

	// Let's go
	auto integrator_thread = std::jthread([&solver]() {
		solver.rock_n_roll();
	});
	ofRunMainLoop();
}
