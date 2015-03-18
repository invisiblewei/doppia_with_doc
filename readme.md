#Overview
the code is a simplification version from the doppia code by rodrigob.

[doppia code](https://bitbucket.org/rodrigob/doppia)

Reserve detection part using integral channels detector and cut the stereo part.

There are still some problems in the CPU version of the code, So I recommend compile with GPU.

Main program is in the folder `/src/applications/objects_detection` 

#Requirements
* Linux (the code can in theory compile and run on windows, but practice has shown this to be a bad idea).
* C++ and CUDA compilation environments properly set. Only gcc 4.5 or superior are supported.
* A GPU with CUDA capability 2.0 or higher (only for objects detection code), and ~200 Mb of free memory (for images of 640x480 pixels).
* All boost libraries.
* Google protocol buffer.
* OpenCv installed (2.4, but code also work with older versions. 3.0 not yet suppoted, but pull requests welcome).
* If speed is your concern, I strongly recommend to compile OpenCv on your machine using CUDA, enabled all relevant SIMD instructions and using -ffast-math -funroll-loops -march=native flags.
* CMake >= 2.4.3 (and knowledge on how to use it).
* Fair amount of patience to get things running.

#How to compile the code ?
##Step 1:
install the requirements

* boost [boost](http://www.boost.org/users/download/)

* Follow C++ Installation - Unix in [Google protocol buffer](https://github.com/google/protobuf/)

* CUDA [cuda-downloads](https://developer.nvidia.com/cuda-downloads)

* OpenCV [opencv](http://opencv.org/)

* CMake `sudo apt-get install cmake`

* install ccmake gui for cmake `sudo apt-get install cmake-curses-gui`

##Step 2: 

* Before trying to compile anything you should also execute (once) generate_protocol_buffer_files.sh to make sure the protocol buffer files match the version installed in your system.
* check the file `common_settings.cmake` and make sure the configuration specfic to your own machine

sm_50 should match your GPU architecture.

`51 set(CUDA_NVCC_FLAGS  "-arch=sm_50" CACHE STRING "CUDA architecture setting" FORCE)`

##Step 3:
1. Go to the application directory `cd src/applications/objects_detection`
1. Make a directory for build and get in `mkdir build && cd build`
1. Run ccmake to config if compile with GPU `ccmake ..`
1. then `cmake .. `and `make -j8`(or `-j10`) to make things faster.
1. If things went well back to the parent directory `cd ../` you should be able to run 

`./build/objects_detection -c eccv2014_face_detection_pascal.config.ini`

#How to test the code?
the config for detector is in `eccv2014_face_detection_pascal.config.ini`

* If you compile cpu only, then change 'method = gpu_channels' to 'method = cpu_channels'
* config pertraining model in `model`
* `score_threshold` threshold for the total score in the detector.
* `x_stride` and `y_stride` stride for sliding window.

#How to use the code?
in `objects_detection_example.cpp` I write a simple example to use and test the detector.

new

`boost::scoped_ptr<doppia::ObjectsDetectionApplication> application_p( new doppia::ObjectsDetectionApplication() );`

init

`bool ok = application_p->init(argc, argv);`

detetor in one image

`application_p->one_step(inputmat);`

get results

`detections_t ds = application_p->getDetections();`

#Read the code
I use `Qt creator` to read the code, it can open cmake project easily.

After run the cmake, you can open the project in Qt creator with `cmake_install.cmake`

Config the run settings in `projects` -> `Build&Run` -> `Run`. Add arguments for config file.
