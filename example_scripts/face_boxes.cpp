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

    auto start_time = std::chrono::steady_clock::now();
    size_t frames_rendered = 0;
    
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
				  cv::Size(100, 100));        // minSize

	for (const auto& face : faces)
	{
	    cv::Point topr(face.x, face.y);
	    cv::Point botl(face.x + face.width, face.y + face.height);
	    cv::rectangle(frame, topr, botl, cv::Scalar(0, 255, 255), 2);
	}

	++frames_rendered;
	auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>
	    (std::chrono::steady_clock::now() - start_time).count();
	double frame_rate = frames_rendered / double(elapsed_time);

	cv::putText(frame, std::to_string(frame_rate), cv::Point(100, 100),
		    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
	
	cv::imshow("c++", frame);
	if( cv::waitKey(1) == 27 ) break; // ESC to end capture.
    }

    return 0;
}
