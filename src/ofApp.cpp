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

	light.setPosition(100, 500, 500);
	black_hole_sphere.setRadius(2.0f * scale_factor);
	black_hole_sphere.setResolution(64);
	black_hole_material.setDiffuseColor(ofColor::orangeRed);
	black_hole_material.setShininess(128);
	camera.tiltDeg(60);
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
	light.enable();
	// ofSetColor(ofColor::orangeRed);
	//ofDrawSphere(0, 0, 2 * scale_factor);

	black_hole_material.begin();
	black_hole_sphere.draw();
	black_hole_material.end();
	// Draw all particles
	particles.draw();

	ofDisableDepthTest();
	light.disable();
	camera.end();

}
