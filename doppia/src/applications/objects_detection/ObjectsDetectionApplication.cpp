
// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include "ObjectsDetectionApplication.hpp"

#include "objects_detection/ObjectsDetectorFactory.hpp"
#include "objects_detection/AbstractObjectsDetector.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"
#include "helpers/ModuleLog.hpp"

#include "helpers/data/DataSequence.hpp"
#include "objects_detection/detections.pb.h"

#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>

#include <boost/format.hpp>

#include <omp.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ObjectsDetectionApplication");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ObjectsDetectionApplication");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ObjectsDetectionApplication");
}

} // end of anonymous namespace

namespace doppia
{

using logging::log;
using namespace std;

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

ObjectsDetectionApplication::ObjectsDetectionApplication()
      :silent_mode(false)
{
    // nothing to do here
    return;
}


ObjectsDetectionApplication::~ObjectsDetectionApplication()
{
    // nothing to do here
    return;
}

bool ObjectsDetectionApplication::init(int argc, char *argv[]){
	program_options::variables_map options_values;
	const bool correctly_parsed = parse_arguments(argc, argv, options_values);
	int return_value = EXIT_SUCCESS;
	if(correctly_parsed)
	{
		this->options = options_values; //store the options for future use
		setup_logging(this->log_file, options_values);

		setup_problem(options_values);

		log_debug() << "setup logging succeed" << std::endl;
		return true;
	}
	return false;
}


template <typename DestView>
DestView convert_Mat_to_Boost_type(const cv::Mat &src)
{

    boost::gil::gil_function_requires<boost::gil::ImageViewConcept<DestView> >();

    typedef typename DestView::value_type::value_type pixel_type;

    return boost::gil::interleaved_view(src.cols, src.rows, (pixel_type *)src.data, src.step);
}

// one_step(view) will process an image given by parent function
void ObjectsDetectionApplication::one_step(const cv::Mat& m_detecting_image)
{

    static int num_iterations = 0;
    static boost::gil::rgb8c_view_t input_view;

    // convert opencv format to boost::gil format
    input_view = convert_Mat_to_Boost_type<boost::gil::rgb8c_view_t>(m_detecting_image);

    double objects_detector_compute_time = 0;
    objects_detector_p->set_image(input_view);
    const double start_objects_detector_compute_wall_time = omp_get_wtime();
    objects_detector_p->compute();
    objects_detector_compute_time = omp_get_wtime() - start_objects_detector_compute_wall_time;

    cout<<"one detector using: "<<objects_detector_compute_time*1000<<"ms in image "<<num_iterations<<endl;

    return;

}

program_options::options_description ObjectsDetectionApplication::get_options_description()
{
    program_options::options_description desc("ObjectsDetectionApplication options");

    const std::string application_name = "objects_detection";

    desc.add_options()

            ("configuration_file,c", program_options::value<string>(),
             "indicate the filename of the configuration .ini file")

            ("log", program_options::value<string>()->default_value(application_name + ".out.txt"),
             "where should the data log be recorded.\n"
             "if 'stdout' is indicated, all the messages will be written to the console\n"
             "if 'none' is indicate, no message will be shown nor recorded");

    desc.add_options()

            ("silent_mode",
             program_options::value<bool>()->default_value(false),
             "if true, no status information will be printed at run time (use this for speed benchmarking)");

    return desc;
}


void ObjectsDetectionApplication::get_all_options_descriptions(program_options::options_description &desc)
{
    // Objects detection options --
    desc.add(ObjectsDetectionApplication::get_options_description());
    desc.add(ObjectsDetectorFactory::get_args_options());

    return;
}


/// helper method called by setup_problem
void ObjectsDetectionApplication::setup_logging(std::ofstream &log_file, const program_options::variables_map &options)
{

    if(log_file.is_open())
    {
        // the logging is already setup
        return;
    }

    logging::get_log().clear(); // we reset previously existing options

    logging::LogRuleSet rules_for_stdout;
    rules_for_stdout.add_rule(logging::EveryMessage, "console");
    logging::get_log().set_console_stream(std::cout, rules_for_stdout);


    const string log_option_value = get_option_value<string>(options, "log");
    // should we also have a log_level option ?

    if(log_option_value != "none")
    {
        logging::LogRuleSet logging_rules;
        logging_rules.add_rule(logging::EveryMessage, "*");

        if(log_option_value == "stdout")
        {
            logging::get_log().add(std::cout, logging_rules);
        }
        else
        {
            // log_option_value != "stdout" and log_option_value != "none"
            if(boost::filesystem::exists(log_option_value))
            {
                printf("Overwriting existing log file %s\n", log_option_value.c_str());
            }
            else
            {
                printf("Creating new log file %s\n", log_option_value.c_str());
            }

            log_file.open(log_option_value.c_str());
            assert(log_file.is_open());

            logging::get_log().add(log_file, logging_rules);
        }
    }
    else
    {
        // log_option_value == "none"
        // nothing else to do, by default logging::log() omits the messages

    }


    const bool silent_mode = get_option_value<bool>(options, "silent_mode");
    if(silent_mode == false)
    {
        // set our own stdout rules --
        logging::LogRuleSet &rules_for_stdout = logging::get_log().console_log().rule_set();
        //rules_for_stdout.add_rule(logging::InfoMessage, "ObjectsDetectionApplication");
#if defined(DEBUG)
        rules_for_stdout.add_rule(logging::DebugMessage, "*"); // we are debugging this application
#else
        rules_for_stdout.add_rule(logging::InfoMessage, "*"); // "production mode"
#endif
    }

    return;
}


void ObjectsDetectionApplication::setup_problem(const program_options::variables_map &options)
{
    // parse the application specific options --
    silent_mode = get_option_value<bool>(options, "silent_mode");

    objects_detector_p.reset(ObjectsDetectorFactory::new_instance(options));

    if(not objects_detector_p)
    {
        throw std::runtime_error("No objects detector selected, this application is pointless without one. Check the value of objects_detector.method");
    }
    return;
}

bool ObjectsDetectionApplication::parse_arguments(int argc, char *argv[], program_options::variables_map &options)
{
    // return values
    const bool arguments_correctly_parsed = true;
    const bool arguments_not_correctly_parsed = !arguments_correctly_parsed;

    program_options::options_description desc("Allowed options");
    desc.add_options()("help", "produces this help message");

    get_all_options_descriptions(desc);

    try
    {
        program_options::command_line_parser parser(argc, argv);
        parser.options(desc);

        const program_options::parsed_options the_parsed_options( parser.run() );

        program_options::store(the_parsed_options, options);
        //program_options::store(program_options::parse_command_line(argc, argv, desc), options);
        program_options::notify(options);
    }
    catch (const std::exception & e)
    {
        cout << "\033[1;31mError parsing the command line options:\033[0m " << e.what () << endl << endl;
        cout << desc << endl;
        throw std::runtime_error("end of game");
        return arguments_not_correctly_parsed;
    }


    if (options.count("help"))
    {
        cout << desc << endl;
        exit(EXIT_SUCCESS);
        return arguments_correctly_parsed;
    }


    // parse the configuration file
    {

        string configuration_filename;

        if(options.count("configuration_file") > 0)
        {
            configuration_filename = get_option_value<std::string>(options, "configuration_file");
        }
        else
        {
            cout << "No configuration file provided. Using command line options only." << std::endl;
        }

        if (configuration_filename.empty() == false)
        {
            boost::filesystem::path configuration_file_path(configuration_filename);
            if(boost::filesystem::exists(configuration_file_path) == false)
            {
                cout << "\033[1;31mCould not find the configuration file:\033[0m "
                     << configuration_file_path << endl;
                return arguments_correctly_parsed;
            }

            try
            {
                fstream configuration_file;
                configuration_file.open(configuration_filename.c_str(), fstream::in);
                program_options::store(program_options::parse_config_file(configuration_file, desc), options);
                configuration_file.close();
            }
            catch (...)
            {
                cout << "\033[1;31mError parsing the configuration file named:\033[0m "
                     << configuration_filename << endl;
                cout << desc << endl;
                throw;
                return arguments_not_correctly_parsed;
            }

            cout << "Parsed the configuration file " << configuration_filename << std::endl;
        }
    }

    return arguments_correctly_parsed;
} // end of parse_arguments

} // end of namespace doppia

//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
