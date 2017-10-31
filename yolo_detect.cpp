#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sys/time.h>
#include <signal.h>

using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Timer;

#define Dtype float

static struct itimerval oldtv;

struct Object {
    cv::Point top_left;
    cv::Point bot_right;
    cv::Rect2d roi;
    cv::UMat sample;
    cv::UMat descriptor;

    std::vector<cv::KeyPoint> keypoint;
    std::vector<cv::DMatch> matchs;

    int id;
    float match_sum;
    float match_avg;

    Object &operator=(const Object &obj) {
        std::cout << "operator = " << std::endl;
        if(this != &obj) {
            this->top_left = obj.top_left;
            this->bot_right = obj.bot_right;
            this->roi = obj.roi;
            this->id = obj.id;
            this->keypoint = obj.keypoint;
            this->matchs = obj.matchs;
            this->match_sum = obj.match_sum;
            this->match_avg = obj.match_avg;
            obj.sample.copyTo(this->sample);
            obj.descriptor.copyTo(this->descriptor);
        }
        return *this;
    }
};
std::vector<struct Object> objects;

struct Target {
    struct Object object;

    cv::Point pt;
    cv::UMat refSample;
};
struct Target target;

struct indicator {
    bool detect;
    bool isInit;
    bool refresh;
    bool click;
    bool isFound;

    indicator() : detect(true), isInit(false), refresh(false),
                  click(false), isFound(false) {}
};
struct indicator flag;

struct SURFDetector {
    cv::Ptr<cv::Feature2D> surf;
    SURFDetector(double hessian = 100.0)
    {
        surf = cv::xfeatures2d::SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher {
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

class Detector {
public:
    Detector(const string& model_file,
             const string& weights_file);

    void Detect(cv::Mat& img);

 private:
  void WrapInputLayer(std::vector<Dtype *> &input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<Dtype *> input_channels);

  shared_ptr<Net<Dtype> > net_;
  cv::Size input_geometry_;
  cv::Size input_newwh_;
  int num_channels_;
};

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
    int count = 0;
    count = Caffe::EnumerateDevices(true);
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
}


Detector::Detector(const string& model_file,
                   const string& weights_file) {
  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
#ifndef CPU_ONLY
    std::cout << "Use GPU with device ID " << gpus[0] << std::endl;
    //Caffe::SetDevices(gpus);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpus[0]);
#endif  // !CPU_ONLY
  } else {
    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);
  }

  /* Load the network. */
  net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Detector::Detect(cv::Mat& img) {
  int w = img.cols;
  int h = img.rows;
  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<Dtype *> input_channels;
  WrapInputLayer(input_channels);

  Preprocess(img, input_channels);

  Timer detect_timer;
  detect_timer.Start();
  double timeUsed;

  net_->Forward();

  detect_timer.Stop();
  timeUsed = detect_timer.MilliSeconds();
  std::cout << "forward time=" << timeUsed <<"ms\n";

  objects.clear();
  /* Copy the output layer to a std::vector */
  Blob<Dtype>* result_blob = net_->output_blobs()[0];
  const Dtype* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  for (int k = 0; k < num_det*7; k+=7) {
	int imgid = (int)result[k+0];
	int classid = (int)result[k+1];
	float confidence = result[k+2];
	int left=0,right,top=0,bot;
    struct Object tmp;
	if(input_newwh_.width==0){
	    left = (int)((result[k+3]-result[k+5]/2.0) * w);
	    right = (int)((result[k+3]+result[k+5]/2.0) * w);
	    top = (int)((result[k+4]-result[k+6]/2.0) * h);
	    bot = (int)((result[k+4]+result[k+6]/2.0) * h);
	}
	else{
        left =  w*(result[k+3] - (input_geometry_.width - input_newwh_.width)/2./input_geometry_.width)*input_geometry_.width / input_newwh_.width;
        top =  h*(result[k+4] - (input_geometry_.height - input_newwh_.height)/2./input_geometry_.height)*input_geometry_.height / input_newwh_.height;
        float boxw = result[k+5]*w*input_geometry_.width/input_newwh_.width;
        float boxh = result[k+6]*h*input_geometry_.height/input_newwh_.height;
        left-=(int)(boxw/2);
        top-=(int)(boxh/2);
        right = (int)(left+boxw);
        bot=(int)(top+boxh);
    }
    if (left < 0)
        left = 0;
    if (right > w-1)
        right = w-1;
    if (top < 0)
        top = 0;
    if (bot > h-1)
        bot = h-1;

    tmp.top_left = cvPoint(left, top);
    tmp.bot_right = cvPoint(right, bot);
    tmp.roi = cv::Rect(left, top, right - left, bot - top);
    tmp.id = classid;
    img(tmp.roi).copyTo(tmp.sample);
    objects.push_back(tmp);
    cv::rectangle(img,cvPoint(left,top),cvPoint(right,bot),cv::Scalar(255, 242, 35));

    std::stringstream ss;
    ss << classid << "/" << confidence;
    std::string  text = ss.str();
    //    cv::putText(img, text, cvPoint(left,top+20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<Dtype *> &input_channels) {
  Blob<Dtype>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  Dtype* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    input_channels.push_back(input_data);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<Dtype *> input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2RGB);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
  else
    sample = img;

  cv::Mat sample_resized;
  int dx=0,dy=0;
  input_newwh_ = input_geometry_;
  if (sample.size() != input_geometry_){
    int netw = input_geometry_.width;
	int neth = input_geometry_.height;
	int width = sample.cols;
	int height = sample.rows;
	if(width!=height){ //if img is not square, must fill the img at center
	    if ((netw*1.0/width) < (neth*1.0/height)){
	        input_newwh_.width= netw;
	        input_newwh_.height = (height * netw)/width;
	    }
	    else{
	        input_newwh_.height = neth;
	        input_newwh_.width = (width * neth)/height;
	    }
	    dx=(netw-input_newwh_.width)/2;
	    dy=(neth-input_newwh_.height)/2;
		cv::resize(sample, sample_resized, input_newwh_);
	}
	else
    	cv::resize(sample, sample_resized, input_geometry_);
  }
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::divide(sample_float, 255.0, sample_normalized);
  if(dx!=0 || dy!=0) {
    for(int i=0;i< num_channels_;i++) {
      for(int pos = 0; pos < input_geometry_.width*input_geometry_.height; ++pos) {
        input_channels[i][pos] = 0.5;
      }
    }
  }

  for( int i = 0; i < input_newwh_.height; i++) {
    for( int j = 0; j < input_newwh_.width; j++) {
      int pos = (i+dy) * input_geometry_.width + j+dx;
      if (num_channels_ == 3) {
        cv::Vec3f pixel = sample_normalized.at<cv::Vec3f>(i, j);
        input_channels[0][pos] = pixel.val[2];
        input_channels[1][pos] = pixel.val[1];
        input_channels[2][pos] = pixel.val[0];  //RGB2BGR
      } else {
        cv::Scalar pixel = sample_normalized.at<float>(i, j);
        input_channels[0][pos] = pixel.val[0];
      }
    }
  }
  if(dx==0 && dy==0)
      input_newwh_.width=0; //clear the flag
}

void onMouse(int event, int x, int y, int flags, void* param)
{
    cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
    switch (event) {
    case CV_EVENT_LBUTTONDOWN:     //鼠标左键按下响应：返回坐标和灰度
        std::cout<<"at("<<x<<","<<y<<")value is:"
                 <<static_cast<int>(im->at<uchar>(cv::Point(x,y)))<<std::endl;
        target.pt = cv::Point(x,y);
        std::cout << "target.pt= " << target.pt << std::endl;
        //        if(flag.detect)
            flag.click = true;
        break;
    }
}

void set_timer(int second)
{
    struct itimerval itv;
    itv.it_interval.tv_sec = second;
    itv.it_interval.tv_usec = 0;
    itv.it_value.tv_sec = second;
    itv.it_value.tv_usec = 0;
    setitimer(ITIMER_REAL, &itv, &oldtv);
}

void signal_handler(int m)
{
    std::cout << "in timer" << std::endl;
}

//void matchFilter(std::vector<cv::DMatch>& m) {
//    sort(m.begin(), m.end());  //筛选匹配点
//    std::vector<cv::DMatch> good_matches;
//    int ptsPairs = std::min(50, (int)(m.size() * 0.35));
//    std::cout << ptsPairs << std::endl;
//    for (int i = 0; i < ptsPairs; ++i) {
//        good_matches.push_back(m[i]);
//    }
//    m.swap(good_matches);
//}




int main(int argc, char** argv) {
    if (argc < 3) {
        return 1;
    }
    std::streambuf* buf = std::cout.rdbuf();
    std::ostream out(buf);
    const string& model_file = argv[1];
    const string& weights_file = argv[2];
    const string& filename = argv[3];
    const string& filenameout = argv[4];

    // Initialize the network.
    Detector detector(model_file, weights_file);

    cv::VideoCapture capture(0);
    //    capture.open("/home/vmt-nuc/Videos/outfile.avi");
    cv::Mat frame, prev_frame;
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    //    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    cv::namedWindow("video", 1);
    cv::setMouseCallback("video", onMouse, reinterpret_cast<void*> (&frame));

    signal(SIGALRM, signal_handler);
    //    set_timer(5);
    int count = -1;
    char buffer[100];
    while(1) {
        std::cout << " in loop: " << ++count << std::endl;
#if 1
        capture >> frame;
#else
        sprintf(buffer, "../img/%04d.jpg", count+1);
        frame = cv::imread(buffer);
        if(frame.empty())
            break;
#endif

//        Timer detect_timer;
//        detect_timer.Start();
//        double timeUsed;

        double start = static_cast<double>(cv::getTickCount());
        if(flag.detect) {
            std::cout << "detect" << std::endl;
            frame.copyTo(prev_frame);
            detector.Detect(frame);

            double end1 = (static_cast<double>(cv::getTickCount()) - start)/cv::getTickFrequency();
            std::cout << "detect time: " << end1*1000 << std::endl;

            if(flag.click) {
                std::cout << "click" << std::endl;
                for(int i = 0; i < objects.size(); ++i) {
                    if(objects[i].roi.contains(target.pt) == true) {
                        target.object = objects[i];
                        if(flag.isInit) {
                            tracker = cv::TrackerKCF::create();
                        }
                        tracker->init(prev_frame, target.object.roi);
                        flag.isInit = true;
                        flag.detect = false;
                        break;
                    }
                }
                flag.click = false;
            }

            double end2 = (static_cast<double>(cv::getTickCount()) - start)/cv::getTickFrequency();
            std::cout << "end2 time: " << end2*1000 << std::endl;

            double refstart = static_cast<double>(cv::getTickCount());
            if(flag.refresh) {
                std::cout << "refresh" << std::endl;
                for(int i = 0; i < objects.size(); ++i) {
                    if(target.object.id == objects[i].id) {
                        std::cout << "roi: " << target.object.roi.size() << std::endl;
                        cv::Mat tmp1;//(target.object.sample.size(),
                                     //target.object.sample.type());
                        cv::Mat tmp2;
                        cv::resize(objects[i].sample, tmp1,
                                   target.object.roi.size(), 0, 0);
                        target.refSample.copyTo(tmp2);
                        if(target.object.roi.x < 0 || target.object.roi.x >640 ||
                           target.object.roi.y < 0 || target.object.roi.y > 480) {
                            target.object.roi.x = std::max(0, (int)(target.object.roi.x));
                            target.object.roi.x = std::min(target.refSample.cols, (int)(target.object.roi.x));
                            target.object.roi.y = std::max(0, (int)(target.object.roi.y));
                            target.object.roi.y = std::max(target.refSample.rows, (int)(target.object.roi.y));
                        }
                        tmp1.copyTo(tmp2(target.object.roi));
                        //                        cv::imshow("replace", tmp2);

                        bool ret = tracker->update(tmp2, target.object.roi);
                        if(ret) {
                            target.object = objects[i];
                            if(flag.isInit) {
                                tracker = cv::TrackerKCF::create();
                            }
                            tracker->init(prev_frame, target.object.roi);
                            flag.refresh = false;
                            flag.detect = false;
                            break;
                        }
                    }
                }
            }
            double refend = (static_cast<double>(cv::getTickCount()) - refstart)/cv::getTickFrequency();
            std::cout << "refend time: " << refend*1000 << std::endl;

            cv::putText(frame, "Detecting", cvPoint(0, 10),
                        cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 0, 255));

            double end3 = (static_cast<double>(cv::getTickCount()) - start)/cv::getTickFrequency();
            std::cout << "end3 time: " << end3*1000 << std::endl;
        } else {
            std::cout << "tracking" << std::endl;
            flag.isFound = tracker->update(frame, target.object.roi);
            if(!flag.isFound) {
                cv::imshow("refsample", target.refSample);
                flag.refresh = true;
                flag.detect = true;
            } else {
                frame.copyTo(target.refSample);
                cv::rectangle(frame, target.object.roi,
                              cv::Scalar( 255, 0, 0 ), 2, 1 );
                cv::putText(frame, "Tracking", cvPoint(0, 10),
                            cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 0, 255));
            }
            //        cv::imshow("last", target.refSample);
        }

//     detect_timer.Stop();
//     timeUsed = detect_timer.MilliSeconds();
//     out << "the first detect time=" << timeUsed <<"ms\n";

        double end = (static_cast<double>(cv::getTickCount()) - start)/cv::getTickFrequency();
        std::cout << "whole time: " << end*1000 << std::endl;

     cv::imwrite(filenameout, frame);
     cv::imshow("video", frame);
     if(!target.object.sample.empty())
         cv::imshow("target", target.object.sample.getMat(cv::ACCESS_FAST));
     if(cv::waitKey(1)==27)
         break;
    }

    return 0;
}
