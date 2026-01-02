#include "particle.h"


void particle::draw() {
	// Update and draw trail
	trail.addVertex(pos.x, pos.y, pos.z);
	ofSetColor(ofColor::white);
	if (trail.size() > 100) 
		trail.removeVertex(0);
	
	trail.draw();

	// Draw particle
	ofSetColor(state == particle_state::in_orbit ? ofColor::chartreuse : ofColor::red);
	ofDrawSphere(pos.x, pos.y, pos.z, 8.0f); 
	
}
