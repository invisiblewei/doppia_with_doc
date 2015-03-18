#include <cstdlib>
#include <iostream>
#include <fstream>
#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>

#include "ObjectsDetectionApplication.hpp"
#include "objects_detection/AbstractObjectsDetector.hpp"

using namespace std;
using namespace boost;

typedef doppia::AbstractObjectsDetector::detections_t detections_t;
typedef doppia::AbstractObjectsDetector::detection_t detection_t;

void draw(cv::Mat &inputmat,vector<cv::Rect> rects,cv::Scalar c)
{
    //show
    for(int i=0;i< rects.size();i++){
        cv::rectangle(inputmat, rects[i],c, 2, 8, 0 );
    }

}
bool same_obj(cv::Rect gt,cv::Rect dect)
{
    double inter = (gt&dect).area();
    double s = inter/(gt.area()+dect.area()-inter);
    return s>0.5;
}
void compare(vector<cv::Rect> gts,vector<cv::Rect> dects,int &tp,int &fp,int &fn)
{
    //int tp=0,fp=0,fn=0;

    for(int i=0;i< gts.size();i++){
        int flag = 0;
        for(int j=0;j< dects.size();j++){
            if(same_obj(gts[i],dects[j])){
                tp++;
                flag++;
                break;
            }
        }
        if(!flag){
            fn++;
        }
    }

    for(int i=0;i< dects.size();i++){
        int flag = 0;
        for(int j=0;j< gts.size();j++){
            if(same_obj(gts[i],dects[j])){
                flag++;
                break;
            }
        }
        if(!flag){
            fp++;
        }
    }
}
// -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;
    string afw_root = "../../../data/afw";
    string afw_lable = afw_root + "/afw_lable.txt";
    string afw_imgs = afw_root + "/testimages/";

    try
    {
        boost::scoped_ptr<doppia::ObjectsDetectionApplication>
        application_p( new doppia::ObjectsDetectionApplication() );

        bool ok = application_p->init(argc, argv);
        ifstream f;
        cout<<afw_lable<<endl;
        f.open(afw_lable.c_str(),fstream::in);
        if(!f){
            cout<<"lable file open error"<<endl;
        }

        if(ok && f){
            cout << "--------------init OK-------------" << endl;
            int tp=0,fp=0,fn=0;
            while(!f.eof()){
                string img_name;
                int face_count;
                f>>img_name;
                string path = afw_imgs+img_name;
                cv::Mat inputmat = cv::imread(path);
                cv::Mat input_s;
                cv::Mat input_show;
                double scale = min(500.0/inputmat.cols,500.0/inputmat.rows);
                if(inputmat.cols<=0 || inputmat.rows<=0){
                    continue;
                }
                cv::resize(inputmat,input_s, cv::Size(0,0), scale,scale,cv::INTER_LINEAR);
                application_p->one_step(input_s);
                detections_t ds = application_p->getDetections();

                vector<cv::Rect> gts;
                f>>face_count;
                for(int i=0;i<face_count;i++){
                    double a,b,c,d;
                    f>>a>>b>>c>>d;
                    cv::Rect r = cv::Rect(a,b,(c-a),(d-b));
                    gts.push_back(r);
                    //cv::rectangle(input_show, r, cv::Scalar( 0, 0, 255 ), 2, 8, 0 );
                }

                vector<cv::Rect> dects;
                for(int i=0;i< ds.size();i++){
                    detection_t::rectangle_t box = ds[i].bounding_box;
                    cv::Rect r = cv::Rect(box.min_corner().x(),
                                           box.min_corner().y(),
                                           box.max_corner().x() - box.min_corner().x(),
                                           box.max_corner().y() - box.min_corner().y());
                    r.x = max(r.x,0)/scale;
                    r.y = max(r.y,0)/scale;
                    r.width = min(r.width,inputmat.cols-r.x-1)/scale;
                    r.height = min(r.height,inputmat.rows-r.y-1)/scale;

                    dects.push_back(r);
                    cout << "--------------score-------------\n" << ds[i].score<<endl;
                }

                compare(gts,dects,tp,fp,fn);
                cout << "---tp---fp---fn---\n" << tp<<","<<fp<<","<<fn<<endl;
                cout << "---precision---recall---\n" << double(tp)/(tp+fp)<<","<<double(tp)/(tp+fn)<<endl;
                bool show = false;
                if(show)
                {
                    inputmat.copyTo(input_show);

                    draw(input_show,dects, cv::Scalar( 0, 255, 0 ));
                    draw(input_show,gts, cv::Scalar( 0, 0, 255 ));

                    imshow( "test", input_show );
                    cv::waitKey(1);
                }
            }
            f.close();
            cout << "---tp---fp---fn---\n" << tp<<","<<fp<<","<<fn<<endl;
            cout << "---precision---recall---\n" << tp/(tp+fp)<<","<<tp/(tp+fn)<<endl;
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


