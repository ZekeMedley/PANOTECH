#include "ThreadPool/ThreadPool.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <queue>
#include <utility>

// The frame's index and its data.
using WorkItem = std::pair<double, cv::Mat>;

// Comparason function for WorkItems
struct WorkItemComparator
{
    // Replaces std::less.
    bool operator()(const WorkItem& lhs, const WorkItem& rhs)
    {
	return lhs.first < rhs.first;
    }
};

int main()
{
    std::priority_queue<WorkItem,
			std::vector<WorkItem>,
			WorkItemComparator> renderQueue;
    std::mutex renderQueueLock;

    cv::CascadeClassifier detector("../../models/haarcascade_frontalface_default.xml");
    
    ThreadPool pool(8);
    
    auto workFn = [&renderQueue, &renderQueueLock, detector](WorkItem wi) mutable
		      {
			  cv::Mat& frame = wi.second;
			  std::vector<cv::Rect> faces;
			  cv::Mat greyscale;
			  
			  cv::cvtColor(frame, greyscale, cv::COLOR_BGR2GRAY);

			  detector.detectMultiScale(greyscale,
			  			    faces,
			  			    1.3,
			  			    3,
			  			    0,
			  			    cv::Size(30, 30));

			  for (const auto& face : faces)
			  {
			      cv::Point top(face.x, face.y);
			      cv::Point bot(face.x + face.width, face.y + face.height);
			      cv::rectangle(frame, top, bot,
			  		    cv::Scalar(0, 255, 255));
			  }
			  
			  renderQueueLock.lock();
			  renderQueue.emplace(std::move(wi));
			  renderQueueLock.unlock();
		      };
    
    cv::VideoCapture cap;

    if ( ! cap.open(0) )
    {
	std::cout << "[error] couldn't open camera stream.\n";
	return 1;
    }

    while ( true )
    {
	cv::Mat frame;
	cap >> frame;

	if ( frame.empty() )
	{
	    std::cout << "[error] failed to read from camera.\n";
	    return 1;
	}
	std::async(workFn, std::make_pair(cap.get(cv::CAP_PROP_POS_FRAMES), std::move(frame)));
//	pool.enqueue(workFn, std::make_pair(cap.get(cv::CAP_PROP_POS_FRAMES), std::move(frame)));
	
	cv::Mat top;
	renderQueueLock.lock();
	
	if ( ! renderQueue.empty() )
	{
	    top = std::move(renderQueue.top().second);
	    renderQueue.pop();
	}

	renderQueueLock.unlock();

	if ( ! top.empty() )
	{
	    cv::imshow("thread pool", top);
	    if( cv::waitKey(1) == 27 ) break;
	}
    }
    return 0;
}
