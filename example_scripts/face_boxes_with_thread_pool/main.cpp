#include "ThreadPool/ThreadPool.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <queue>
#include <utility>
#include <chrono>

inline std::vector<cv::Rect> getFaces(const cv::Mat frame)
{
    static cv::CascadeClassifier detector("../../models/haarcascade_frontalface_default.xml");

    std::vector<cv::Rect> faces;
    cv::Mat greyscale;

    cv::cvtColor(frame, greyscale, cv::COLOR_BGR2GRAY);
    detector.detectMultiScale(greyscale,
			      faces,
			      1.3,
			      3,
			      0,
			      cv::Size(100, 100));

    return faces;
}

int main()
{
    std::queue<cv::Rect> renderQueue;
    std::mutex renderQueueLock;

    cv::CascadeClassifier detector("../../models/haarcascade_frontalface_default.xml");
    
    auto workFn = [&](const cv::Mat frame) mutable
		      {
			  renderQueueLock.lock();
			  for (const auto& r : getFaces(frame))
			      renderQueue.push(std::move(r));
			  renderQueueLock.unlock();
		      };
    
    cv::VideoCapture cap;
    if ( ! cap.open(0) )
    {
	std::cout << "[error] couldn't open camera stream.\n";
	return 1;
    }

    auto start_time = std::chrono::steady_clock::now();
    size_t frames_rendered = 0;
    
    while ( true )
    {
	cv::Mat frame;
	cap >> frame;

	if ( frame.empty() )
	{
	    std::cout << "[error] failed to read from camera.\n";
	    return 1;
	}
	std::async(workFn, frame);

	// Pop all the rectangles out of the render queue.
	std::vector<cv::Rect> toRender;
	renderQueueLock.lock();
	if ( ! renderQueue.empty() )
	{
	    while ( ! renderQueue.empty() )
	    {
		toRender.emplace_back(std::move(renderQueue.front()));
		renderQueue.pop();
	    }
	}
	renderQueueLock.unlock();

	// Unlock the queue and draw all of them.
	for ( const auto& face : toRender )
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
	cv::imshow("thread pool", frame);
	if( cv::waitKey(1) == 27 ) break;
    }
    return 0;
}
