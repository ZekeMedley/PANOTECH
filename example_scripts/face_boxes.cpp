#include <opencv2/opencv.hpp>

#include <iostream>

int main()
{
    cv::VideoCapture cap;

    if ( ! cap.open(0) )
    {
	std::cout << "[error] couldn't open camera stream.\n";
	return 1;
    }

    cv::CascadeClassifier detector("../models/haarcascade_frontalface_default.xml");

    cv::Mat frame;
    cv::Mat greyscale;
    
    while ( true )
    {
	cap >> frame;
	
	if ( frame.empty() )
	{
	    std::cout << "[error] failed to read from camera.\n";
	    return 1;
	}

	std::vector<cv::Rect> faces;

	cv::cvtColor(frame, greyscale, cv::COLOR_BGR2GRAY);
	// cv example does this, our python does not.
//	cv::equalizeHist(greyscale, greyscale);

	detector.detectMultiScale(greyscale,
				  faces,
				  1.3,                        // scaleFactor
				  3,                          // minNeighbors
				  0,                          // flags
				  cv::Size(30, 30));          // minSize

	for (const auto& face : faces)
	{
	    cv::Point topr(face.x, face.y);
	    cv::Point botl(face.x + face.width, face.y + face.height);
	    cv::rectangle(frame, topr, botl, cv::Scalar(0, 255, 255));
	}

	cv::imshow("c++", frame);
	if( cv::waitKey(1) == 27 ) break; // ESC to end capture.
    }

    return 0;
}
