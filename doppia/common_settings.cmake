# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# This file is intended to be included by other cmake files (see src/applications/*/CMakeLists.txt)

# ----------------------------------------------------------------------

#site_name(HOSTNAME)

# Use "gcc -v -Q -march=native -O3 test.c -o test" to see which options the compiler actualy uses
# using -march=native will include all sse options available on the given machine (-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2, etc...)
# also using -march=native will imply -mtune=native
# Thus the optimization flags below should work great on all machines
# (O3 is already added by CMAKE_CXX_FLAGS_RELEASE)
set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native")
#set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native -freciprocal-math -funsafe-math-optimizations -fassociative-math -ffinite-math-only -fcx-limited-range")  # cheap -Ofast copy
#set(OPT_CXX_FLAGS "-ffast-math -funroll-loops -march=native") # disabled OpenMp, just for testing
#set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops") # disable native compilation, so we can profile with older versions of valgrind/callgrind

# enable link time optimization
# http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
#set(OPT_CXX_FLAGS "${OPT_CXX_FLAGS} -flto")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
#set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -flto")

message(STATUS "change the default optimisation options in common_settings.cmake")
option(USE_GPU "Should the GPU be used ?" False)
if(USE_GPU)
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)
  set(local_CUDA_CUT_INCLUDE_DIRS "/usr/local/cuda/include")
  set(local_CUDA_CUT_LIBRARY_DIRS "/usr/local/cuda/lib")
  set(local_CUDA_LIB_DIR "/usr/lib")
  set(local_CUDA_LIB "/usr/lib/libcuda.so")
  set(cuda_LIBS "cuda")
  set(CUDA_NVCC_EXECUTABLE  /usr/local/cuda/bin/nvcc )
  set(CUDA_SDK_ROOT_DIR  /usr/local/cuda)
  
  message(STATUS "architecture sm_50 for nvidia GTX750")
  
# CUDA architecture setting: going with all of them (up to CUDA 5.5 compatible).
# For the latest architecture, you need to install CUDA >= 6.0 and uncomment
# the *_50 lines below.
		
  set(CUDA_NVCC_FLAGS  "-arch=sm_50" CACHE STRING "CUDA architecture setting" FORCE)
endif()
set(google_perftools_LIBS tcmalloc_and_profiler)


# ----------------------------------------------------------------------
# enable compilation for shared libraries
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fpic" CACHE STRING "nvcc flags" FORCE)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  # enable cuda debug information, to use with cuda-dbg
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G" CACHE STRING "nvcc flags" FORCE)
else()
# FIXME disabled only for testing
#  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math"  CACHE STRING "nvcc flags" FORCE) # speed up host and device code
endif()

# ----------------------------------------------------------------------
# set default compilation flags and default build

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # add debug information, even in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS ${OPT_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")


if(USE_GPU)
  add_definitions(-DUSE_GPU)
endif(USE_GPU)


# set default cmake build type (None Debug Release RelWithDebInfo MinSizeRel)
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE "Release" )
endif()

# ----------------------------------------------------------------------
