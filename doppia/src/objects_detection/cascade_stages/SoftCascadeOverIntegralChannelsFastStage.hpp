#ifndef DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTSTAGE_HPP
#define DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTSTAGE_HPP

//#include "SoftCascadeOverIntegralChannelsStage.hpp" // for IntegralChannelsFeature

#include "helpers/geometry.hpp"
#include <boost/cstdint.hpp>

#if defined(__CUDACC__)
#include <host_defines.h>
#endif


namespace doppia {

// forward declarations
//class SoftCascadeOverIntegralChannelsStage;
//class DecisionStump;

/// efficient and compact version of the equivalent doppia_protobuf message
class IntegralChannelsFeature
{
public:
    typedef boost::int16_t coordinate_t;

    typedef doppia::geometry::point_xy<coordinate_t> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    /// integral channel index
    boost::uint8_t channel_index;

    /// box over which compute the integral
    rectangle_t box;
};


class DecisionStump
{
public:

    typedef IntegralChannelsFeature feature_t;
    feature_t feature;

    /// thresholding the corresponding rectangle provides the a weak binary classifier
    float feature_threshold;

    /// if true will evaluate (feature >= threshold),
    /// if false, will evaluate (feature < threshold)
    bool larger_than_threshold;

    inline bool operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        if(feature_value >= feature_threshold)
        {
            return larger_than_threshold;
        }
        else
        {
            return not larger_than_threshold;
        }
    }
};

class Level2DecisionTree
{
public:

    typedef DecisionStump::feature_t feature_t;

    DecisionStump level1_node;
    DecisionStump level2_true_node;
    DecisionStump level2_false_node;
};

/// just like DecisionStump, but assuming larger_than_threshold == true
class SimpleDecisionStump
{
public:

    typedef IntegralChannelsFeature feature_t;
    feature_t feature;

    /// thresholding the corresponding rectangle provides a weak binary classifier
    float feature_threshold;

    inline bool operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        return (feature_value >= feature_threshold);
    }
};


class DecisionStumpWithWeights: public SimpleDecisionStump
{
public:

    typedef SimpleDecisionStump::feature_t feature_t;
    typedef feature_t::rectangle_t rectangle_t;

    __host__ __device__
    DecisionStumpWithWeights();
    DecisionStumpWithWeights(const DecisionStump &stump, const float feature_weight);
    //~DecisionStumpWithWeights();

    /// strong_classifier_score = sum_over_features(weight of weak_classifier_output)
    float weight_true_leaf, weight_false_leaf;

    inline float operator()(const float feature_value) const
    {
        // uses >= to be consistent with Markus Mathias code
        return (feature_value >= feature_threshold)? weight_true_leaf : weight_false_leaf;
    }
};

/// Defined in the header to make CUDA happy
inline
DecisionStumpWithWeights::DecisionStumpWithWeights()
{
    // default constructor does nothing
    return;
}


class Level2DecisionTreeWithWeights
{
public:
    typedef SimpleDecisionStump::feature_t feature_t;
    typedef feature_t::rectangle_t rectangle_t;

    SimpleDecisionStump level1_node;
    DecisionStumpWithWeights level2_true_node;
    DecisionStumpWithWeights level2_false_node;


    /// Bounding box for the features included in all the nodes,
    /// this box is used to do fast image border condition checking
    rectangle_t bounding_box;

    /// compute the bounding box based on the current nodes values
    void compute_bounding_box();
};


/// This class contains the same data as SoftCascadeOverIntegralChannelsStage,
/// however the data is stored a different format that allows
/// faster detections at runtime (both in cpu and gpu versions)
/// @see SoftCascadeOverIntegralChannelsModel.hpp
class //__align__(16)
SoftCascadeOverIntegralChannelsFastStage
{
public:

    __host__  __device__
    SoftCascadeOverIntegralChannelsFastStage();

    // no destructor definition to avoid problem when doing copies inside the GPU
    //~SoftCascadeOverIntegralChannelsFastStage();


    //typedef DecisionStumpWithWeights weak_classifier_t;
    typedef Level2DecisionTreeWithWeights weak_classifier_t;

    weak_classifier_t weak_classifier;

    /// if (strong_classifier_score < cascade_threshold) answer is "not this class"
    float cascade_threshold;

    /// helper function when doing partial objects detection (borders of the image)
    const weak_classifier_t::rectangle_t &get_bounding_box() const;
};


/// defined in the header to make CUDA happy
inline
SoftCascadeOverIntegralChannelsFastStage::SoftCascadeOverIntegralChannelsFastStage()
{
    // default constructor does nothing
    return;
}


inline
const SoftCascadeOverIntegralChannelsFastStage::weak_classifier_t::rectangle_t &
SoftCascadeOverIntegralChannelsFastStage::get_bounding_box() const
{
    return weak_classifier.bounding_box;
}




} // end of namespace doppia

#endif // DOPPIA_SOFTCASCADEOVERINTEGRALCHANNELSFASTSTAGE_HPP
