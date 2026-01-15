#include "ofApp.h"
#include "utils.h"
#include "message.h"
#include "particle.h"
#include "message_queue.h"

extern message_queue<message> data_queue;

void ofApp::setup() {
	ofSetVerticalSync(true);
	//ofSetFrameRate(2000);
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

	// Update particle data
	particles.x = std::move(message.xs);
	particles.y = std::move(message.ys);
	particles.z = std::move(message.zs);
	particles.rad = std::move(message.radii);
	particles.states = std::move(message.states);
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
