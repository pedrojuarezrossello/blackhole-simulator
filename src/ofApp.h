#pragma once

#include "ofMain.h"
#include "particle.h"

class ofApp : public ofBaseApp {

	public:
		void setup();
		void update();
		void draw();

	private:
		
		ofEasyCam camera;
		particle_set particles;
		ofRectangle viewMain;

};
