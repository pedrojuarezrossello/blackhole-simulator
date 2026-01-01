#include "ofApp.h"
#include "utils.h"
#include "message.h"
#include "particle.h"
#include "message_queue.h"
#include <ranges>
#include <algorithm>
#include <execution>

extern message_queue<message> data_queue;

constexpr float scale_factor = 30.0f;

void ofApp::setup() {
	ofSetVerticalSync(true);
	ofEnableDepthTest();
	ofSetCircleResolution(64);
	ofBackground(0, 0, 0);

	light.setPosition(100, 500, 500);
	black_hole_sphere.setRadius(2.0f * scale_factor);
	black_hole_sphere.setResolution(64);
	black_hole_material.setDiffuseColor(ofColor::orangeRed);
	black_hole_material.setShininess(128);
	camera.tiltDeg(60);

}

void ofApp::update() {
	// Wait for an update from queue
	auto message = data_queue.pop();
	
	if (particles.particles.size() < message.xs.size()) [[unlikely]] 
		particles.particles = std::vector<particle>(message.xs.size());

	// Update each particle data
	size_t N = message.xs.size();
	 for (size_t i = 0; i < N; ++i) {
		particles.particles[i].pos.x = message.xs[i] * scale_factor;
		particles.particles[i].pos.y = message.ys[i] * scale_factor;
		particles.particles[i].pos.z = message.zs[i] * scale_factor;
		particles.particles[i].state = message.states[i];
	}
}

void ofApp::draw(){
	
	camera.begin();
	light.enable();

	// Draw black hole
	black_hole_material.begin();
	black_hole_sphere.draw();
	black_hole_material.end();

	// Draw all particles
	particles.draw();

	ofDisableDepthTest();
	light.disable();
	camera.end();
}
