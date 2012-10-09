#include "ofxCvFaceRec.h"

ofxCvFaceRec::ofxCvFaceRec() {
    nTrainFaces               = 0; // the number of training images
    nEigens                   = 0; // the number of eigenvalues
    projectedTrainFaceMat = 0; // projected training faces
    faceImgArr        = 0; // array of face images
    eigenVectArr      = 0; // eigenvectors
    pAvgTrainImg       = 0; // the average image
    eigenValMat        = 0   ; // eigenvalues
    personNumTruthMat = 0; // array of person numbers
    trained = false;
}

ofxCvFaceRec::~ofxCvFaceRec() {
}

void ofxCvFaceRec::learn() {
    int i, offset;	

    // load training data
    nTrainFaces = loadFaceImgArray("train.txt");
    if( nTrainFaces < 2 )
    {
    printf("Need 2 or more training faces\n"
            "Input file contains only %d\n", nTrainFaces);
    return;
    }

    // do PCA on the training faces
    doPCA();

    // project the training images onto the PCA subspace
    projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
    offset = projectedTrainFaceMat->step / sizeof(float);
    for(i=0; i<nTrainFaces; i++)
    {
        //int offset = i * nEigens;
        cvEigenDecomposite(
            faceImgArr[i],
            nEigens,
            eigenVectArr,
            0, 0,
            pAvgTrainImg,
            projectedTrainFaceMat->data.fl + i*nEigens);
            //projectedTrainFaceMat->data.fl + i*offset);
    }
    // store the recognition data as an xml file
    storeTrainingData();

    trained=true;
}

void ofxCvFaceRec::mask(ofxCvGrayscaleImage img) {


};

int ofxCvFaceRec::recognize(ofxCvGrayscaleImage img) {
    int i, nearest, truth, iNearest;

    //nTestFaces  = 0;         // the number of test images
    CvMat * trainPersonNumMat = 0;  // the person numbers during training
    float * projectedTestFace = 0;
    char Buf[50];

    // load the saved training data
    if(!trained) {
        if( !loadTrainingData( &trainPersonNumMat ) ) return -1;
    };

    // mask the test image

    // project the test images onto the PCA subspace
    projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

    // project the test image onto the PCA subspace
    cvEigenDecomposite(
        img.getCvImage(),
        nEigens,
        eigenVectArr,
        0, 0,
        pAvgTrainImg,
        projectedTestFace);

    iNearest = findNearestNeighbor(projectedTestFace);

//    printf("weights: ");
//    for(i=0; i<nEigens; i++)
//        printf("%f ", projectedTestFace[i]);
//    printf("\n");

    return iNearest;
}

void ofxCvFaceRec::doPCA() {
    int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainFaces-1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

    ofxCvFloatImage eig;

    for(i=0; i<nEigens; i++) {
        // add eigenVectArr[i] to eigens
        //eig.setFromPixels(eigenVectArr[i]->imageData, eigenVectArr[i]->width, eigenVectArr[i]->height);
        eig.allocate(faceImgSize.width, faceImgSize.height);

        eig=eigenVectArr[i];
        eig.convertToRange(0., 255.);
        eigens.push_back(eig);
        eig.clear();
    };

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

void ofxCvFaceRec::storeTrainingData() {
    CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	string outfile = ofToDataPath("facedata.xml");

	fileStorage = cvOpenFileStorage(outfile.c_str(), 0, CV_STORAGE_WRITE );

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}

int ofxCvFaceRec::loadTrainingData(CvMat ** pTrainPersonNumMat) {
  CvFileStorage * fileStorage;
	int i;

    string outfile = ofToDataPath("facedata.xml");

	// create a file-storage interface
	fileStorage = cvOpenFileStorage(outfile.c_str() , 0, CV_STORAGE_READ );
	if( !fileStorage )
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );

	return 1;
}

int ofxCvFaceRec::findNearestNeighbor(float * projectedTestFace) {
      //double leastDistSq = 1e12;
    char Buf[100];
    //Out->clear();
    leastDistSq = DBL_MAX;
    int i, iTrain, iNearest = 0;

    for(iTrain=0; iTrain<nTrainFaces; iTrain++)
    {
      double distSq=0;
      for(i=0; i<nEigens; i++)
      {
        float d_i =	projectedTestFace[i] -
            projectedTrainFaceMat->data.fl[iTrain*nEigens + i];

            distSq += d_i*d_i; // Euclidean

      }
      //sprintf(Buf,"%03d  ->  %f",iTrain+1,distSq);
      //Out->add(Buf);

      if(distSq < leastDistSq){
         leastDistSq = distSq;
         iNearest = iTrain;
      }
    }

    //printf("leastDistSq: %03d -> %f\n", iNearest, leastDistSq);

    return iNearest;
}

int ofxCvFaceRec::loadFaceImgArray(char * filename) {
    FILE * imgListFile = 0;
    char imgFilename[512];
    char Buf[512];
    int iFace, nFaces=0;
	ofImage img;

    string fileName = ofToDataPath(filename);

    // open the input file
    if( !(imgListFile = fopen(fileName.c_str(), "r")) )
    {
        fprintf(stderr, "Can\'t open file %s\n", fileName.c_str());
        //return 0;
    } else {

        // count the number of faces
        while( fgets(imgFilename, 512, imgListFile) ) {
            ++nFaces;
        };
        rewind(imgListFile);
    }

    // allocate the face-image array and person number matrix
    faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
    personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

    if(!faces.empty()) faces.clear();

    // store the face images in an array
    for(iFace=0; iFace<nFaces; iFace++)
    {
        // read image name from file
        fgets(imgFilename, 512, imgListFile);
        printf("imgFilename: %s", imgFilename);

        string temp = imgFilename;
        temp = "faces/"+temp;

        string inImage = ofToDataPath(temp);
        inImage = inImage.substr(0, inImage.length()-1);

        //strcpy(imgFilename, ofToDataPath(files[iFace]).c_str());

        *(personNumTruthMat->data.i+iFace)= iFace+1;

        if (img.loadImage(inImage) == false) {
            printf("Can\'t load image from %s\n", inImage.c_str());
            return 0;
        }

        img.resize(PCA_WIDTH, PCA_HEIGHT);
        img.update();

        ofxCvGrayscaleImage gray;
        ofxCvColorImage color;

        switch(img.type) {
            case OF_IMAGE_GRAYSCALE:
                gray.allocate(img.width, img.height);
                gray = img.getPixels();//color;
                break;
            case OF_IMAGE_COLOR:
            case OF_IMAGE_COLOR_ALPHA:
                color.allocate(img.width, img.height);
                color = img.getPixels();
                gray.allocate(img.width, img.height);
                gray = color;
                break;
            case OF_IMAGE_UNDEFINED:
                printf("Image is of unknown type, %s\n", imgFilename);
                return 0;
        };

        //img.draw(200+iFace*100, 200);//, 50, 50);
        //color.draw(200+iFace*100, 300);//, 50, 50);
        //gray.draw(200+iFace*100, 400);//, 50, 50);

        // add gray image to array of training images
        //gray.contrastStretch();
        faces.push_back(gray);
        color_faces.push_back(color);

        IplImage *im8 = gray.getCvImage();
        faceImgArr[iFace] = cvCloneImage(im8);
    }

    fclose(imgListFile);

    return nFaces;
}

void ofxCvFaceRec::draw(int x, int y)
{
    drawFaces(x, y);
    drawEigens(x, y+faces[0].height+25);
}

void ofxCvFaceRec::drawFaces(int x, int y)
{
    int i;

	for(i = 0; i < faces.size(); i++)
        faces[i].draw((x+i*faces[i].width), y);
}

void ofxCvFaceRec::drawFaces(int x, int y, int width)
{
    int i;

	for(i = 0; i < faces.size(); i++)
        faces[i].draw((x+i*(width/faces.size())), y, width/faces.size(), faces[i].height*(width/faces.size())/faces[i].width);
}

void ofxCvFaceRec::drawHilight(int pnum, int x, int y, int width)
{
    // set properties
    ofNoFill();
    ofSetColor(((0xFF) << pnum)%0xFFFFFF);
    ofSetLineWidth(4.0);

    ofRect(x+pnum*(width/faces.size()), y, width/faces.size(), faces[pnum].height*(width/faces.size())/faces[pnum].width);

    // reset
    ofSetColor(255, 255, 255, 255);

}

void ofxCvFaceRec::drawEigens(int x, int y)
{
    int i;

    for(i=0; i<nEigens; i++)
        eigens[i].draw((x+i*eigens[i].width), y);

}

void ofxCvFaceRec::drawEigens(int x, int y, int width)
{
    int i;

    for(i=0; i<nEigens; i++)
        eigens[i].draw((x+i*(width/nEigens)), y, width/nEigens, eigens[i].height*(width/nEigens)/eigens[i].width);
}

void ofxCvFaceRec::drawPerson(int pnum, int x, int y, int w, int h)
{
    if((pnum < 0) || (pnum > faces.size())) return;

    faces[pnum].draw(x, y, w, h);
}

void ofxCvFaceRec::drawPerson(int pnum, int x, int y)
{
    if((pnum < 0) || (pnum > faces.size())) return;

    faces[pnum].draw(x, y);
}

void ofxCvFaceRec::drawColorPerson(int pnum, int x, int y, int w, int h)
{
    if((pnum < 0) || (pnum > color_faces.size())) return;

    color_faces[pnum].draw(x, y, w, h);
}

void ofxCvFaceRec::drawColorPerson(int pnum, int x, int y)
{
    if((pnum < 0) || (pnum > color_faces.size())) return;

    color_faces[pnum].draw(x, y);
}

unsigned char* ofxCvFaceRec::getPersonPixels(int pnum) {
    if((pnum < 0) || (pnum > color_faces.size())) return NULL;

    return color_faces[pnum].getPixels();
}

//void ofxCvFaceRec::drawComposite(int x, int y)
//{
//
//}
