---
layout: single
title:  "Face detection in OpenCv"
date:   2018-12-09 21:37:36 +0530
categories: OpenCv
---
Image processing is a method to convert an image into digital form and perform some operations on it, in order to get an enhanced image or to extract some useful information from it.
It is among rapidly growing technologies today, with its applications in various aspects of a business.
Image Processing includes four steps-
1. Image Capture
2. Image Sharpening and Restoration
3. Image retreival - Seek for the image of interest.
4. Image Recognition - Distinguish the objects in an image

In our Project, We have made a Face detection Model. In this , we studied about processing of image using Open Source Library, OpenCV (Open Computer Vision) . We first did Image processing on our digital image by using in-built OpenCv computer Algorithms. Then, to detect the face, we passed our image to the in-built OpenCV classifier , LBP based Classifier. This pretrained model works on our image , and detects the face. Then, this face area is further preproccessed , and smoothed to get us just our face in a  new Window.

The libraries , we used are -

1. Core Module -> Its a compact module defining basic data structures, including the dense multi- dimensional array Mat and basic functions used by all other modules.

2. Image Processing -> Its an image processing module that includes linear and non- linear image filtering, geometrical image transformations (resize, affine and perspective warping , generic table - based remapping), color space conversion, histograms etc.

3. features2d - > It has salient feature detectors, descriptors and descriptor matches.

4. Highgui -> Its an easy to use interface to simple UI capabilities.

5. Video I/O -> Its an easy to use interface to video capturing and video codecs.

6. objdetect -> It helps in detection of objects and instances of the predefined classes.for example, faces, eyes, mugs etc.

We captured the videostream from our webcam. The, we stored each frame of our video stream into a mat object. We converted the Mat color object into gray object. The, we scaled our gray image to a constant width of 320. For a width of 320, we also scaled our height appropriately. This was done for our cascade classifier to run better. As , we are using pre- built classifier, it works better on small width images, but not too small. So, we chose a medium Size image.
Now coming to the classifier part which will detect face:  
In 2001 Viola and Jones invented the Haar based cascade classifier for object detection and in 2002 in was improved by Heinhart and Maydt. The result is an object detector that is both fast and reliable (detects approx. 95% of frontal face correctly).
It works not only for frontal faces but also side view faces, eyes, mouth etc. however in our project we have used only frontal face classifier.
We have used LBP based detector as it is several times faster than Haar based detector.
Haar based detector: it performs detection by doing comparison like; if we look at frontal face, region with eye should be darker than forehead and cheeks, and region with mouth darker than cheeks.
But, for LBP based face detector It performs comparison but uses histograms of pixel intensity comparison such as edges, corners and flat regions.
Then, face preprocessing is performed to give final result Separate histogram equalization for left and right sides.  
This process standardizes the brightness and contrast on both the left and right hand sides of face independently.
Smoothening - It reduces the image noise using a bilateral filter.  


{% highlight c++ %}
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <algorithm>

using namespace std;

using namespace cv;

int main(){
Mat colorimage;
Mat grayimage;
//Mat hsvimage;

VideoCapture capture(0);

if(!capture.isOpened()){
cout<<"Failed to open webcam";
}
capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);

char key =0;
while(key !='q'){

//Capture a frame and store it in imag variable
capture >>colorimage;

CascadeClassifier faceDetector;
  try{
     faceDetector.load("../Desktop/OpenCV_tutorial/OpenCV/opencv-3.3.0/data/lbpcascades/lbpcascade_frontalface.xml");
     namedWindow("Color Image",CV_WINDOW_NORMAL);
     imshow("Color Image", colorimage);

      Mat gray;
       if(colorimage.channels()==3)
       cvtColor(colorimage,gray,CV_BGR2GRAY);
       else if (colorimage.channels()==4){
         cvtColor(colorimage,gray,CV_BGR2GRAY);
         }
          else
          gray = colorimage;

     //Gray Image - Converting Color To Gray Image
     //namedWindow("Gray Image",CV_WINDOW_NORMAL);
    // imshow("GrayScale Image",gray);


const int Detect_width = 320;
Mat smallImg;
//cout <<gray.cols<<" "<<"320"<<endl;
// proportional way reduction of pic
float scale = gray.cols/(float)Detect_width;
//cout<<scale <<" ";
if(gray.cols > Detect_width){
int ScaledHeight = cvRound(gray.rows/scale);
resize(gray, smallImg, Size(Detect_width, ScaledHeight));
}
else
smallImg = gray;

// Resized Image -> Re sizing Image
//namedWindow("Re-Sized Image",CV_WINDOW_NORMAL);
//imshow("Re-Sized image",smallImg);

//Standardize the brightness and contrast
Mat equalizedImg;
equalizeHist(smallImg, equalizedImg);
//namedWindow("Histogram Image",CV_WINDOW_NORMAL);
//imshow(" Histogram Image ", equalizedImg);

int flags = CASCADE_SCALE_IMAGE;
Size minFeatureSize(20,20);
float searchScaleFactor = 1.1f;
int minNeighbors = 4;
//int scale = 1;
//Detect Objects in the small grayscale image
vector<Rect> faces;
faceDetector.detectMultiScale( equalizedImg , faces,searchScaleFactor,
       minNeighbors, flags, minFeatureSize);

cout<<"the number of faces is "<<faces.size()<<endl;
int scaledWidth = Detect_width;
//Enlarge the results if the image was temporarily shrunk.
cout<<scale<<endl;
if( equalizedImg.cols > scaledWidth){  //used the width of the frame
  for( int i=0;i < (int)faces.size(); i++){
          faces[i].x = cvRound( faces[i].x * scale);
          faces[i].y = cvRound( faces[i].y * scale);
          faces[i].width = cvRound( faces[i].width * scale);
          faces[i].height = cvRound( faces[i].height *scale);
          }
          }

//If the object is on a border, keep it on the image.
for(int i=0;i<(int)faces.size();i++){
 if(faces[i].x <0)
faces[i].x =0;
if (faces[i].y<0)
faces[i].y =0;
if (faces[i].x + faces[i].width > equalizedImg.cols)
faces[i].x = equalizedImg.cols - faces[i].width;
if(faces[i].y + faces[i].height >  equalizedImg.rows)
faces[i].y = equalizedImg.rows - faces[i].height;
}
int g = (int)faces.size();
Mat  faceImg;
faceImg = equalizedImg;
Rect rect_crop ;
 for (int  i = 0; i < g; i++)
        {
			Rect rect = faces[i];
            rectangle(equalizedImg, faces[i], Scalar(0,255,255),5);
            rect_crop =  Rect(rect.x,rect.y,rect.width,rect.height);

        }
        Mat image_roi =  Mat(faceImg , rect_crop);
faceImg = image_roi;
namedWindow("Detected Frontal Face",CV_WINDOW_NORMAL);
if(equalizedImg.rows >0 && equalizedImg.cols >0)
  imshow("Detected Frontal Face",equalizedImg);

 /*
  vector<string> algorithms;
Algorithm::getList(algorithms);
int kl = (int)algorithms.size();
cout <<"Algorithms :";
cout<< kl<<endl;
for(int i=0;i<kl;i++){
	cout<< algorithms[i] <<endl;
}
*/
int w = faceImg.cols;
int h = faceImg.rows;
Mat wholeFace;
equalizeHist(faceImg, wholeFace);
int midX = w/2;
Mat leftSide = faceImg(Rect(0,0,midX,h));
Mat rightSide = faceImg(Rect(midX, 0, w-midX,h));

equalizeHist(leftSide,leftSide);
equalizeHist(rightSide, rightSide);
if(leftSide.rows >0 && leftSide.cols >0)
imshow("Left side face",leftSide);
if(rightSide.rows >0 && rightSide.cols >0)
imshow("Right Side face",rightSide);
/*
if (leftSide.width >0 && rightSide.width >0 && rightSide.height >0 && leftSide.height >0){
namedWindow("Left side face",CV_WINDOW_NORMAL);
imshow("Left side face",leftSide);
namedWindow("Right Side face",CV_WINDOW_NORMAL);
imshow("Right Side face",rightSide);
}
*/
// To Combine the three images together
for (int y =0;y <h;y++){
	for(int x=0;x<w;x++){
		int v;
		if(x<w/4){
			v = leftSide.at<uchar>(y,x);
		}
		else if(x < w*1/2){
			int lv = leftSide.at<uchar>(y,x);
			int wv = wholeFace.at<uchar>(y,x);
			//Bland more of a face
			float f = (x-w*1/4)/(float)(w/4);
			v = cvRound((1.0f - f)*lv + (f)*wv);
		}
		else if(x < w*3/4){
			int rv = rightSide.at<uchar>(y,x-midX);
			int wv = wholeFace.at<uchar>(y,x);
			float f = (x-w*2/4)/(float)(w/4);
			v = cvRound((1.0f-f)*wv + (f)*rv);
		}
		else{
		v = rightSide.at<uchar>(y,x-midX);
	}
	faceImg.at<uchar>(y,x) = v;
	}
}
if (faceImg.rows>0 && faceImg.cols>0)
imshow("final face",faceImg);
// Smoothing
Mat filtered = Mat(faceImg.size(),CV_8U);
bilateralFilter(faceImg,filtered, 0,20.0,2.0);
if(filtered.cols >0 && filtered.rows >0)
imshow("Filtered",filtered);


}catch(Exception e){
	continue;
cerr<<"ERROR : Couldn't load face Detection ";
cerr<<"OOPSS!!"<<endl;
exit(1);

}
key = waitKey(25);
}
return 0;
}
{% endhighlight %}

[code-gh]:   https://github.com/aditi-mishra/Face-Detection
