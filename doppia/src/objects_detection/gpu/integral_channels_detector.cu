#include "integral_channels_detector.cu.hpp"

#include "helpers/gpu/cuda_safe_call.hpp"

#include "cudatemplates/array.hpp"
#include "cudatemplates/symbol.hpp"
#include "cudatemplates/copy.hpp"

#include <cudatemplates/hostmemoryheap.hpp>

#include <boost/cstdint.hpp>

#include <stdexcept>



namespace {

/// small helper function that computes "static_cast<int>(ceil(static_cast<float>(total)/grain))", but faster
static inline int div_up(const int total, const int grain)
{
    return (total + grain - 1) / grain;
}

} // end of anonymous namespace


namespace doppia {
namespace objects_detection {

using namespace cv::gpu;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 1>::Texture gpu_integral_channels_1d_texture_t;
gpu_integral_channels_1d_texture_t integral_channels_1d_texture;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 2>::Texture gpu_integral_channels_2d_texture_t;
gpu_integral_channels_2d_texture_t integral_channels_2d_texture;

// global variable to switch from using 1d textures to using 2d textures
// On visics-gt680r 1d texture runs at ~4.8 Hz; 2d texture runs at ~4.4 Hz
//const bool use_2d_texture = false;
const bool use_2d_texture = true;

/// this method will do zero memory checks, the user is responsible of avoiding out of memory accesses
inline
__device__
float get_feature_value_global_memory(const IntegralChannelsFeature &feature,
                                      const int x, const int y,
                                      const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride,
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    const gpu_integral_channels_t::Type
            a = integral_channels.data[top_left_index],
            b = integral_channels.data[top_right_index],
            c = integral_channels.data[bottom_right_index],
            d = integral_channels.data[bottom_left_index];

    const float feature_value = a +c -b -d;

    return feature_value;
}


inline
__device__
float get_feature_value_tex1d(const IntegralChannelsFeature &feature,
                              const int x, const int y,
                              const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride;
    const size_t
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_1d_texture_t &tex = integral_channels_1d_texture;
#define tex integral_channels_1d_texture
    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    // tex1Dfetch should be used to access linear memory (not text1D)
    const gpu_integral_channels_t::Type
    //const float
            a = tex1Dfetch(tex, top_left_index),
            b = tex1Dfetch(tex, top_right_index),
            c = tex1Dfetch(tex, bottom_right_index),
            d = tex1Dfetch(tex, bottom_left_index);
#undef tex


    const float feature_value = a +c -b -d;

    return feature_value;
}

template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_3d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    const size_t integral_channels_height = integral_channels.size[1];
    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    const gpu_integral_channels_t::Type
    //const float
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_2d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    const size_t integral_channels_height = integral_channels.height; // magic trick !

    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    //const float
    const gpu_integral_channels_t::Type
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType, bool should_use_2d_texture>
inline
__device__
float get_feature_value(const FeatureType &feature,
                        const int x, const int y,
                        const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    // default implementation (hopefully optimized by the compiler)
    if (should_use_2d_texture)
    {
        return get_feature_value_tex2d(feature, x, y, integral_channels);
    }
    else
    {
        //return get_feature_value_global_memory(feature, x, y, integral_channels);
        return get_feature_value_tex1d(feature, x, y, integral_channels);
    }

    //return 0;
}

inline
__device__
bool evaluate_decision_stump(const DecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    if(feature_value >= stump.feature_threshold)
    {
        return stump.larger_than_threshold;
    }
    else
    {
        return not stump.larger_than_threshold;
    }
}


inline
__device__
bool evaluate_decision_stump(const SimpleDecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold);
}


inline
__device__
float evaluate_decision_stump(const DecisionStumpWithWeights &stump,
                              const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold)? stump.weight_true_leaf : stump.weight_false_leaf;
}

template<typename CascadeStageType>
inline
__device__
void update_detection_score(
        const int x, const int y,
        const CascadeStageType &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_score)
{
    const typename CascadeStageType::weak_classifier_t &weak_classifier = stage.weak_classifier;
    typedef typename CascadeStageType::weak_classifier_t::feature_t feature_t;

    // level 1 nodes evaluation returns a boolean value,
    // level 2 nodes evaluation returns directly the float value to add to the score

    const float level1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(
                weak_classifier.level1_node.feature, x, y, integral_channels);

    // On preliminary versions,
    // evaluating the level2 features inside the if/else
    // runs slightly faster than evaluating all of them beforehand; 4.35 Hz vs 4.55 Hz)
    // on the fastest version (50 Hz or more) evaluating all three features is best

    const bool use_if_else = false;
    if(not use_if_else)
    { // this version is faster

        const float level2_true_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_true_node.feature, x, y, integral_channels);

        const float level2_false_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_false_node.feature, x, y, integral_channels);

        current_score +=
                (evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value)) ?
                    evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value) :
                    evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
    }
    else
    {
        if(evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value))
        {
            const float level2_true_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_true_node.feature, x, y, integral_channels);

            current_score += evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value);
        }
        else
        {
            const float level2_false_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_false_node.feature, x, y, integral_channels);

            current_score +=evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
        }

    }

    return;
}

#if defined(BOOTSTRAPPING_LIB)
const bool use_hardcoded_cascade = true;
//const int hardcoded_cascade_start_stage = 100; // this is probably good enough
const int hardcoded_cascade_start_stage = 100; // to be on the safe side
const float hardcoded_cascade_threshold = -5;
//const int hardcoded_cascade_start_stage = 500; // this is conservative
#else
// FIXME these should be templated options, selected at runtime
//const bool use_hardcoded_cascade = true;
const bool use_hardcoded_cascade = true;
//const int hardcoded_cascade_start_stage = 100;
//const int hardcoded_cascade_start_stage = 250; // same as during bootstrapping
const int hardcoded_cascade_start_stage = 100;
//const int hardcoded_cascade_start_stage = 1000;

// will break if (detection_score < hardcoded_cascade_threshold)
//const float hardcoded_cascade_threshold = 0;
//const float hardcoded_cascade_threshold = -1;
const float hardcoded_cascade_threshold = -0.03;
#endif

/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector

template <typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const int search_range_width, const int search_range_height,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        dev_mem_ptr_step_float_t detection_scores)
{
    const int
            x = blockIdx.x * blockDim.x + threadIdx.x,
            //y = blockIdx.y;
            y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x >= search_range_width) or ( y >= search_range_height))
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0; //detection_scores_row[x];

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index = 0; stage_index < cascade_length; stage_index += 1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        if(detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    float* detection_scores_row = detection_scores.ptr(y);
    detection_scores_row[x] = detection_score; // save the updated score
    //detection_scores_row[x] = cascade_length; // just for debugging
    return;
}


/// type int because atomicAdd does not support size_t
__device__ int num_gpu_detections[1];
Cuda::Symbol<int, 1> num_gpu_detections_symbol(Cuda::Size<1>(1), num_gpu_detections);
int num_detections_int;
Cuda::HostMemoryReference1D<int> num_detections_host_ref(1, &num_detections_int);


void move_num_detections_from_cpu_to_gpu(size_t &num_detections)
{ // move num_detections from CPU to GPU --
    num_detections_int = static_cast<int>(num_detections);
    Cuda::copy(num_gpu_detections_symbol, num_detections_host_ref);
    return;
}

void move_num_detections_from_gpu_to_cpu(size_t &num_detections)
{ // move (updated) num_detections from GPU to CPU
    Cuda::copy(num_detections_host_ref, num_gpu_detections_symbol);
    if(num_detections_int < static_cast<int>(num_detections))
    {
        throw std::runtime_error("Something went terribly wrong when updating the number of gpu detections");
    }
    num_detections = static_cast<size_t>(num_detections_int);
    return;
}



template<typename ScaleType>
inline
__device__
void add_detection(
        gpu_detections_t::KernelData &gpu_detections,
        const int x, const int y, const ScaleType scale_index,
        const float detection_score)
{
    gpu_detection_t detection;
    detection.scale_index = static_cast<boost::int16_t>(scale_index);
    detection.x = static_cast<boost::int16_t>(x);
    detection.y = static_cast<boost::int16_t>(y);
    detection.score = detection_score;

    const size_t detection_index = atomicAdd(num_gpu_detections, 1);
    if(detection_index < gpu_detections.size[0])
    {
        // copy the detection into the global memory
        gpu_detections.data[detection_index] = detection;
    }
    else
    {
        // we drop out of range detections
    }

    return;
}


/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector
template <typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const gpu_scale_datum_t scale_datum,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{

    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    //const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            //delta_y = blockIdx.y;
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;


    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0;

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index = 0; stage_index < cascade_length; stage_index += 1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        if(detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    // >= to be consistent with Markus's code
    if(detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);
    }

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 1d binding
void bind_integral_channels_to_1d_texture(gpu_integral_channels_t &integral_channels)
{

    //integral_channels_texture.filterMode = cudaFilterModeLinear; // linear interpolation of the values
    integral_channels_1d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 1d data,
    // and then have ad-hoc texture access in the kernel
    // (if interpolation is needed, will need to do a different 2d data hack
    const cudaChannelFormatDesc texture_channel_description = \
            cudaCreateChannelDesc<gpu_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
            or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // FIXME add this 3D to 1D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture(0, integral_channels_1d_texture, integral_channels.getBuffer(),
                               texture_channel_description, integral_channels.getBytes()));

    cuda_safe_call( cudaGetLastError() );

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_3d_integral_channels_t &integral_channels)
{ // 3d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_2d_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 2d data,
    // and then have ad-hoc texture access in the kernel
    const cudaChannelFormatDesc texture_channel_description = cudaCreateChannelDesc<gpu_3d_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
            or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc seems to have failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // Layout.size is width, height, num_channels
    const size_t
            integral_channels_width = integral_channels.getLayout().size[0],
            integral_channels_height = integral_channels.getLayout().size[1],
            num_integral_channels = integral_channels.getLayout().size[2],
            //channel_stride = integral_channels.getLayout().stride[1],
            row_stride = integral_channels.getLayout().stride[0],
            pitch_in_bytes = row_stride * sizeof(gpu_3d_integral_channels_t::Type);

    if(false)
    {
        printf("image width/height == %zi, %zi; row_stride == %zi\n",
               integral_channels.getLayout().size[0], integral_channels.getLayout().size[1],
               integral_channels.getLayout().stride[0]);

        printf("integral_channels size / channel_stride == %.3f\n",
               integral_channels.getLayout().stride[2] / float(integral_channels.getLayout().stride[1]) );
    }

    // FIXME add this 3D to 2D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture2D(0, integral_channels_2d_texture, integral_channels.getBuffer(),
                                 texture_channel_description,
                                 integral_channels_width, integral_channels_height*num_integral_channels,
                                 pitch_in_bytes));

    cuda_safe_call( cudaGetLastError() );
    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_2d_integral_channels_t &integral_channels)
{ // 2d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values

    // integral_channels_height == (channel_height*num_channels) + 1
    // 2d to 2d binding
    integral_channels.bindTexture(integral_channels_2d_texture);

    cuda_safe_call( cudaGetLastError() );
    return;
}



void bind_integral_channels_texture(gpu_integral_channels_t &integral_channels)
{
    if(use_2d_texture)
    {
        bind_integral_channels_to_2d_texture(integral_channels);
    }
    else
    {
        bind_integral_channels_to_1d_texture(integral_channels);
    }

    return;
}


void unbind_integral_channels_texture()
{

    if(use_2d_texture)
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_2d_texture) );
    }
    else
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_1d_texture) );
    }
    cuda_safe_call( cudaGetLastError() );

    return;
}

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// this method directly adds elements into the gpu_detections vector
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_impl(gpu_integral_channels_t &integral_channels,
                                     const size_t search_range_index,
                                     const doppia::ScaleData &scale_data,
                                     GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
                                     const float score_threshold,
                                     gpu_detections_t& gpu_detections,
                                     size_t &num_detections)
{

    const doppia::DetectorSearchRange &search_range = scale_data.scaled_search_range;

    gpu_scale_datum_t scale_datum;
    {
        scale_datum.search_range.min_corner().x( search_range.min_x );
        scale_datum.search_range.min_corner().y( search_range.min_y );
        scale_datum.search_range.max_corner().x( search_range.max_x );
        scale_datum.search_range.max_corner().y( search_range.max_y );

        scale_datum.stride.x( scale_data.stride.x() );
        scale_datum.stride.y( scale_data.stride.y() );
    }


    //if((search_range.detection_window_scale/search_range.range_scaling) == 1)
    if(false)
    {
        printf("Occlusion type == %i\n", search_range.detector_occlusion_type);
        printf("integral_channels_detector_impl search range min (x,y) == (%.3f, %.3f), max (x,y) == (%.3f, %.3f)\n",
               (search_range.min_x/search_range.range_scaling),
               (search_range.min_y/search_range.range_scaling),
               (search_range.max_x/search_range.range_scaling),
               (search_range.max_y/search_range.range_scaling));

        //throw std::runtime_error("Stopping everything so you can debug");
    }

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    //const int nthreads = 320; // we optimize for images of width 640 pixel
    //dim3 block_dimensions(32, 10);
    //const int block_x = std::max(4, width/5), block_y = std::max(1, 256/block_x);

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            block_x = 16,
            block_y = num_threads / block_x;
    dim3 block_dimensions(block_x, block_y);

    const int
            width = search_range.max_x - search_range.min_x,
            height = search_range.max_y - search_range.min_y;

    if((width <= 0) or (height <= 0))
    { // nothing to be done
        // num_detections is left unchanged
        return;
    }

    dim3 grid_dimensions(div_up(width, block_dimensions.x),
                         div_up(height, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    integral_channels_detector_kernel
            <CascadeStageType>
            <<<grid_dimensions, block_dimensions>>>
                                                  (scale_datum,
                                                   integral_channels,
                                                   search_range_index,
                                                   detection_cascade_per_scale,
                                                   score_threshold,
                                                   gpu_detections);

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::ScaleData &scale_data,
                                gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, scale_data, detection_cascade_per_scale,
                                    score_threshold, gpu_detections, num_detections);
    return;
}

} // end of namespace objects_detection
} // end of namespace doppia

