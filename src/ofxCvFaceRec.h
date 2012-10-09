#pragma once

/*

Face Recognition class, implementing PCA, adapted from "Seeing with
OpenCV Part 5":
http://www.cognotics.com/opencv/servo_2007_series/part_5/index.html
and bluekid's demo app
http://derindelimavi.blogspot.com/2008/05/yz-tanma-2.html

It is intended to work in concert with the ofxCvHaarFinder class,
and as of OF 006, requires the ofxOpenCv-for-0061 modifications to
fix some problems of the ofxCvFloatImage.

ofxCvHaarFinder:
http://www.openframeworks.cc/forum/viewtopic.php?f=10&t=2006
ofxOpenCv:
http://www.openframeworks.cc/forum/viewtopic.php?f=10&t=1967

robert twomey

*/

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "cvaux.h"

#define DROP_LOW 5
#define DROP_HIGH 3

const int PCA_HEIGHT = 150;
const int PCA_WIDTH = 150;

const int PCA_HEIGHT_2 = PCA_HEIGHT/2;
const int PCA_WIDTH_2 = PCA_WIDTH/2;

class ofxCvFaceRec {
public:

	ofxCvFaceRec();
	~ofxCvFaceRec();

    // public methods
    int loadFaceImgArray(char * filename);
    void learn();
    bool isTrained() { return trained; };

    int recognize(ofxCvGrayscaleImage img);

    void draw(int x, int y);
    void drawFaces(int x, int y);
    void drawFaces(int x, int y, int width);
    void drawEigens(int x, int y);
    void drawEigens(int x, int y, int width);

    void drawHilight(int pnum, int x, int y, int width);

    void drawPerson(int pnum, int x, int y);
    void drawPerson(int pnum, int x, int y, int w, int h);
    void drawColorPerson(int pnum, int x, int y);
    void drawColorPerson(int pnum, int x, int y, int w, int h);

    unsigned char* getPersonPixels(int pnum);
    double getLeastDistSq() { return leastDistSq; };

    int numPeople() { return nTrainFaces; };

protected:

    void mask(ofxCvGrayscaleImage img);

    void doPCA();
    void storeTrainingData();
    int loadTrainingData(CvMat ** pTrainPersonNumMat);
    int findNearestNeighbor(float * projectedTestFace);

    int nTrainFaces               ; // the number of training images
    int nEigens                   ; // the number of eigenvalues
    CvMat * projectedTrainFaceMat ; // projected training faces
    IplImage ** faceImgArr        ; // array of face images
    IplImage ** eigenVectArr      ; // eigenvectors
    IplImage * pAvgTrainImg       ; // the average image
    CvMat * eigenValMat           ; // eigenvalues
    CvMat    *  personNumTruthMat ; // array of person numbers

    double leastDistSq;

    vector<ofxCvGrayscaleImage> faces;
    vector<ofxCvColorImage> color_faces;
    vector<ofxCvFloatImage> eigens;

    vector<ofImage> faceSprites;

    bool trained;

};
