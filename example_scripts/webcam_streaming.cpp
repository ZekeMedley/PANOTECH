// To compile you'll need opencv installed. On a Mac, the easiest way
// to do that is with ``brew install opencv``. You might want to
// google around about installing gcc without building it if you want
// to save time.

// clang++ $(pkg-config --cflags --libs opencv4)
// -std=c++17 webcam_streaming.cpp -o webcam ./webcam to run

#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    cv::VideoCapture cap;

    if( ! cap.open(0) )
        return 0;

    for(;;)
    {
	cv::Mat frame;
	cap >> frame;
	if( frame.empty() ) break; // end of video stream
	imshow("this is you, smile! :)", frame);
	if( cv::waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}
