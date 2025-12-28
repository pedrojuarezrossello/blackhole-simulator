#include "ofApp.h"
#include "utils.h"
#include "message.h"
#include "particle.h"
#include "message_queue.h"
#include <ranges>
#include <algorithm>
#include <execution>

extern message_queue<message<N>> data_queue;

constexpr float scale_factor = 30.0f;

void ofApp::setup() {
	ofSetVerticalSync(true);
	// this uses depth information for occlusion
	// rather than always drawing things on top of each other
	ofEnableDepthTest();
	ofSetCircleResolution(64);
	ofBackground(0, 0, 0);
	camera.setTarget(particles);
	//camera.enableInertia();
}

void ofApp::update() {
	// Wait for an update from queue
	auto message = data_queue.wait_and_pop();

	// Update each particle data
	 for (size_t i = 0; i < N; ++i) {
		//this->particles.particles[i].pos.x = ofGetWidth() / 2 + message.xs[i] * 15.0f;
		//this->particles.particles[i].pos.y = ofGetHeight() / 2 - message.ys[i] * 15.0f;
		this->particles.particles[i].pos.x = message.xs[i] * scale_factor;
		this->particles.particles[i].pos.y = message.ys[i] * scale_factor;
		this->particles.particles[i].pos.z = message.zs[i] * scale_factor;
		this->particles.particles[i].state = message.states[i];
	}
}

void ofApp::draw(){
	camera.begin();
	ofSetColor(ofColor::orangeRed);
//	ofFill();
	ofDrawSphere(0, 0, 2 * scale_factor);
	
	// Draw all particles
	particles.draw();

	//ofDrawGrid(20, 10, true, true, true, true);
	ofDisableDepthTest();
	camera.end();
	
	//std::ranges::for_each(particles, &particle::draw);
}
