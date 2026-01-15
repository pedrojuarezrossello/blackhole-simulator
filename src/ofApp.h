#pragma once

#include "ofMain.h"
#include "particle.h"

class ofApp : public ofBaseApp {

	public:
	ofApp(size_t N, float a, kerr _)
		: camera(ofEasyCam{})
		, particles(N, a, _)
		, light(ofLight {})
		, black_hole_sphere(ofSpherePrimitive {})
		, black_hole_material(ofMaterial {}) { }

	ofApp(size_t N, float bm, schwarzschild _)
		: camera(ofEasyCam {})
		, particles(N, bm, _)
		, light(ofLight {})
		, black_hole_sphere(ofSpherePrimitive {})
		, black_hole_material(ofMaterial {}) { }

		void setup();
		void update();
		void draw();
	private:
		ofEasyCam camera;
		particle_set particles;
		ofLight light;
		ofSpherePrimitive black_hole_sphere;
		ofMaterial black_hole_material;
};
