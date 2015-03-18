#ifndef ObjectsDetectionApplication_HPP
#define ObjectsDetectionApplication_HPP

#include "objects_detection/AbstractObjectsDetector.hpp"

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

namespace doppia_protobuf {
class Detections;
}


namespace doppia
{

using namespace std;
using namespace  boost;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

 // forward declarations
class ObjectsDetectionGui;
class AbstractObjectsDetector;
template<typename DataType> class DataSequence;
class ImagesFromDirectory;

class ObjectsDetectionApplication
{
	typedef doppia::AbstractObjectsDetector::detections_t detections_t;
    typedef doppia::AbstractObjectsDetector::detection_t detection_t;

protected:


public:
    typedef DataSequence<doppia_protobuf::Detections> DetectionsDataSequence;
	bool parse_arguments(int argc, char *argv[], program_options::variables_map &options);
    static program_options::options_description get_options_description();

    ObjectsDetectionApplication();
    ~ObjectsDetectionApplication();

	bool init(int argc, char *argv[]);
	void one_step(const cv::Mat& inputmat);
    detections_t getDetections(){return objects_detector_p->get_detections();}

protected:
	
    void get_all_options_descriptions(program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const program_options::variables_map &options);
    void setup_problem(const program_options::variables_map &options);

    scoped_ptr<AbstractObjectsDetector> objects_detector_p;
    program_options::variables_map options;
    bool silent_mode;
	std::ofstream log_file;
	
};


} // end of namespace doppia

#endif // ObjectsDetectionApplication_HPP
