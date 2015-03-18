#include <cstdlib>
#include <iostream>

#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>

#include "ObjectsDetectionApplication.hpp"
#include "objects_detection/AbstractObjectsDetector.hpp"

using namespace std;
using namespace boost;

typedef doppia::AbstractObjectsDetector::detections_t detections_t;
typedef doppia::AbstractObjectsDetector::detection_t detection_t;

// -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;

    try
    {
        boost::scoped_ptr<doppia::ObjectsDetectionApplication>
        application_p( new doppia::ObjectsDetectionApplication() );

		bool ok = application_p->init(argc, argv);
		if(ok){
            cout << "--------------init OK-------------" << endl;
            cv::Mat inputmat = cv::imread("../../../data/sample_test_images/pascal_faces/BJJJCDBGIJ20140307105633.png");
			application_p->one_step(inputmat);
            detections_t ds = application_p->getDetections();

            //show
            for(int i=0;i< ds.size();i++){
                detection_t::rectangle_t box = ds[i].bounding_box;
                cv::Rect r = cv::Rect(box.min_corner().x(),
                                       box.min_corner().y(),
                                       box.max_corner().x() - box.min_corner().x(),
                                       box.max_corner().y() - box.min_corner().y());
                r.x = max(r.x,0);
                r.y = max(r.y,0);
                r.width = min(r.width,inputmat.cols-r.x-1);
                r.height = min(r.height,inputmat.rows-r.y-1);

                cv::rectangle(inputmat, r, cv::Scalar( 255, 0, 255 ), 2, 8, 0 );
                cout << "--------------score-------------\n" << ds[i].score<<endl;
            }
            imshow( "test", inputmat );
            cv::waitKey(0);
		}

		cout << "End of game, have a nice day." << endl;
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        cout << "\033[1;31mAn unknown exception was raised\033[0m " << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}


