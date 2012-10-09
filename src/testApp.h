#pragma once

#include "ofMain.h"
#include "ofxCvHaarFinder.h"
#include "ofxCvFaceRec.h"

#define SCALE 2
#define TEST_DIV 2
#define CAM_WIDTH 640
#define CAM_HEIGHT 480

class testApp : public ofBaseApp{
	public:
		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
        void calcFaceSprites();

		ofImage img;
		ofImage test_image;
		ofImage bgImage;
		ofImage mask;
		unsigned char *mask_pixels;

		ofxCvHaarFinder finder;
        ofxCvFaceRec rec;

        ofVideoGrabber 		vidGrabber;
		ofTexture			videoTexture;
		int 				camWidth;
		int 				camHeight;

		ofImage face;
        ofxCvColorImage color;
        ofxCvGrayscaleImage gray;

        vector <ofImage> faces;

    private:
        // vars to toggle onscreen display
        bool showEigens;
        bool showFaces;
        bool showExtracted;
        bool showTest;
        bool showLeastSq;
        bool bgSubtract;
        bool showClock;
};
