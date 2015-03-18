#include "SoftCascadeOverIntegralChannelsModel.hpp"

#include "detector_model.pb.h"

#include "cascade_stages/check_stages_and_range_visitor.hpp"

#include "helpers/Log.hpp"

#include <boost/foreach.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/format.hpp>
//#include <boost/geometry/geometry.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ostream>
#include <vector>
#include <cmath>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::vector<float> upscaling_factors;
std::vector<float> downscaling_factors;

} // end of anonymous namespace


namespace doppia {

// we alias all the cascade types, to make it easier to define the variant visitors
typedef SoftCascadeOverIntegralChannelsModel::variant_stages_t variant_stages_t;

typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t fast_stage_t;
typedef SoftCascadeOverIntegralChannelsModel::fast_stages_t fast_stages_t;

void normalize_maximum_detection_score(const doppia_protobuf::DetectorModel &model,
                                       doppia_protobuf::DetectorModel &rescaled_model)
{
    rescaled_model.CopyFrom(model);

    const float desired_max_score = 1;

    log_info() << "Detector model is being normalized to have maximum detection score == "
               << desired_max_score << std::endl;

    doppia_protobuf::SoftCascadeOverIntegralChannelsModel &soft_cascade =
            *(rescaled_model.mutable_soft_cascade_model());

    float weights_sum = 0;
    for(int stage_index = 0; stage_index < soft_cascade.stages_size(); stage_index += 1)
{
        weights_sum += soft_cascade.stages(stage_index).weight();
    } // end for "for each cascade stage"

    log_debug() << "The Detector model out of training had a max score of " << weights_sum << std::endl;

    const float weight_scaling_factor = desired_max_score / weights_sum;

    for(int stage_index = 0; stage_index < soft_cascade.stages_size(); stage_index += 1)
    {
        doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage =
                *(soft_cascade.mutable_stages(stage_index));

        stage.set_weight(stage.weight() * weight_scaling_factor);

        if(stage.cascade_threshold() > -1E5)
        { // rescale the cascade threshold if it has a non-absurd value
            stage.set_cascade_threshold(stage.cascade_threshold() * weight_scaling_factor);
    }
    } // end for "for each cascade stage"

    return;
}

SoftCascadeOverIntegralChannelsModel::SoftCascadeOverIntegralChannelsModel(const doppia_protobuf::DetectorModel &model)
{

    if(model.has_detector_name())
    {
        log_info() << "Parsing model " << model.detector_name() << std::endl;
        //semantic_category = model.semantic_category();
    }

    if(model.detector_type() != doppia_protobuf::DetectorModel::SoftCascadeOverIntegralChannels)
    {
        throw std::runtime_error("Received model is not of the expected type SoftCascadeOverIntegralChannels");
    }

    if(model.has_soft_cascade_model() == false)
    {
        throw std::runtime_error("The model content does not match the model type");
    }

    doppia_protobuf::DetectorModel normalized_model;

    normalize_maximum_detection_score(model, normalized_model);

    set_stages_from_model(normalized_model.soft_cascade_model());

    shrinking_factor = normalized_model.soft_cascade_model().shrinking_factor();

    if(normalized_model.has_scale())
    {
        scale = normalized_model.scale();
    }
    else
    {
        // default scale is 1 ("I am the canonical scale")
        scale = 1.0;
    }

    model_window_size.x(model.model_window_size().x());
    model_window_size.y(model.model_window_size().y());

    if(normalized_model.has_semantic_category())
    {
        semantic_category = model.semantic_category();
    }
    else
    {
        log_warning() << "No semantic category found in model, assuming a pedestrian detector" << std::endl;
        semantic_category = "/m/017r8p"; // see http://www.freebase.com/m/017r8p
    }

    if(normalized_model.has_object_window())
    {
        const doppia_protobuf::Box &the_object_window = model.object_window();
        object_window.min_corner().x(the_object_window.min_corner().x());
        object_window.min_corner().y(the_object_window.min_corner().y());
        object_window.max_corner().x(the_object_window.max_corner().x());
        object_window.max_corner().y(the_object_window.max_corner().y());
    }

    if(model.has_occlusion_level())
    {
        occlusion_level = model.occlusion_level();
    }
    else
    {
        occlusion_level = 0; // by default we assume no occlusion
    }

    if(model.has_occlusion_type())
    {
        switch(model.occlusion_type())
        {
        case doppia_protobuf::DetectorModel::LeftOcclusion:
            occlusion_type = LeftOcclusion;
            break;

        case doppia_protobuf::DetectorModel::RightOcclusion:
            occlusion_type = RightOcclusion;
            break;

        case doppia_protobuf::DetectorModel::BottomOcclusion:
            occlusion_type = BottomOcclusion;
            break;

        case doppia_protobuf::DetectorModel::TopOcclusion:
            occlusion_type = TopOcclusion;
            throw std::runtime_error("Current code base does not support top occlusions");
            break;

        default:
            throw std::runtime_error("SoftCascadeOverIntegralChannelsModel received a model with an unknown occlusion type");
        }
    }
    else
    {
        occlusion_type = NoOcclusion; // any default value is fine
    }


    if((occlusion_type == LeftOcclusion) or (occlusion_type == TopOcclusion))
    { // left and top occlusion need to modify the model itself, so that the features "stick" to the top left corner

        shift_stages_by_occlusion_level();
    }

    if((occlusion_level > 0) and (occlusion_type == NoOcclusion))
    {
        throw std::runtime_error("Detector model should not have occlusion_type == NoOcclusion and, "
                                 "at the same time, occlusion_level > 0");
    }

    sanity_check();

    return;
}



SoftCascadeOverIntegralChannelsModel::~SoftCascadeOverIntegralChannelsModel()
{
    // nothing to do here
    return;
}


template<typename DecisionStumpType>
void set_decision_stump_feature(const doppia_protobuf::IntegralChannelDecisionStump &stump_data,
                                DecisionStumpType &stump)
{

    stump.feature.channel_index = stump_data.feature().channel_index();

    const doppia_protobuf::Box &box_data = stump_data.feature().box();
    IntegralChannelsFeature::rectangle_t &box = stump.feature.box;
    box.min_corner().x(box_data.min_corner().x());
    box.min_corner().y(box_data.min_corner().y());
    box.max_corner().x(box_data.max_corner().x());
    box.max_corner().y(box_data.max_corner().y());

    stump.feature_threshold = stump_data.feature_threshold();

    //const float box_area = boost::geometry::area(box);
    const float box_area =
            (box.max_corner().x() - box.min_corner().x())*(box.max_corner().y() - box.min_corner().y());


    if(box_area == 0)
    {
        log_warning() << "feature.box min_x " << box.min_corner().x() << std::endl;
        log_warning() << "feature.box min_y " << box.min_corner().y() << std::endl;
        log_warning() << "feature.box max_x " << box.max_corner().x() << std::endl;
        log_warning() << "feature.box max_y " << box.max_corner().y() << std::endl;
        throw std::runtime_error("One of the input features has area == 0");
    }


    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data, DecisionStump &stump)
{
    set_decision_stump_feature(stump_data, stump);
    stump.larger_than_threshold = stump_data.larger_than_threshold();
    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data, SimpleDecisionStump &stump)
{
    set_decision_stump_feature(stump_data, stump);

    // nothing else todo here

    // no need to check this, handled at the set_weak_classifier level
    if(false and stump_data.larger_than_threshold() == false)
    {
        throw std::runtime_error("SimpleDecisionStump was set using stump_data.larger_than_threshold == false, "
                                 "we expected it to be true; code needs to be changed to handle this case.");
    }
    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data,
                        const float stage_weight,
                        DecisionStumpWithWeights &stump)
{
    set_decision_stump_feature(stump_data, stump);

    if(stump_data.has_true_leaf_weight() and stump_data.has_false_leaf_weight())
    {

        if(stump_data.larger_than_threshold())
        {
            stump.weight_true_leaf = stump_data.true_leaf_weight();
            stump.weight_false_leaf = stump_data.false_leaf_weight();
        }
        else
        {
            stump.weight_true_leaf = stump_data.false_leaf_weight();
            stump.weight_false_leaf = stump_data.true_leaf_weight();
        }

        if(stage_weight != 1.0)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsStage the leaves have true/false weights, "
                                     "but the stage weight is different than 1.0. "
                                     "This case is not currently handled.");
        }
    }
    else
    {
        if(stump_data.larger_than_threshold())
        {
            stump.weight_true_leaf = stage_weight;
            stump.weight_false_leaf = -stage_weight;
        }
        else
        {
            stump.weight_true_leaf = -stage_weight;
            stump.weight_false_leaf = stage_weight;
        }
    }

    return;
}

void set_weak_classifier(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data,
                         SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage)
{

    if(boost::is_same<SoftCascadeOverIntegralChannelsFastStage::weak_classifier_t, Level2DecisionTreeWithWeights>::value)
    {
        if(stage_data.feature_type() != doppia_protobuf::SoftCascadeOverIntegralChannelsStage::Level2DecisionTree)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage contains a feature_type != Level2DecisionTree");
        }

        if(stage_data.has_level2_decision_tree() == false)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage feature_type == Level2DecisionTree but "
                                     "no level2_decision_tree has been set");
        }

        if(stage_data.level2_decision_tree().nodes().size() != 3)
        {
            log_error() << "stage_data.level2_decision_tree.nodes().size() == "
                        << stage_data.level2_decision_tree().nodes().size() << ", not 3" << std::endl;
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage level2_decision_tree does not contain 3 nodes");
        }

        const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode
                *level_1_node_p = NULL, *level2_true_node_p = NULL, *level2_false_node_p = NULL;
        for(int i=0; i < stage_data.level2_decision_tree().nodes().size(); i+=1)
        {
            const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode &node = stage_data.level2_decision_tree().nodes(i);
            if(node.id() == node.parent_id())
            {
                level_1_node_p =  &node;
            }
            else if(node.has_parent_value() and node.parent_value() == true)
            {
                level2_true_node_p = &node;
            }
            else if(node.has_parent_value() and node.parent_value() == false)
            {
                level2_false_node_p = &node;
            }
            else
            {
                // we skip weird nodes
            }
        }

        if(level_1_node_p == NULL)
        {
            throw std::runtime_error("Could not find the parent of the decision tree");
        }

        if(level2_true_node_p == NULL or level2_false_node_p == NULL)
        {
            throw std::runtime_error("Could not find one of the children nodes of the decision tree");
        }

        if(level_1_node_p->decision_stump().larger_than_threshold() == false)
        {
            std::swap(level2_true_node_p, level2_false_node_p);
        }

        const float stage_weight = stage_data.weight();
        set_decision_stump(level_1_node_p->decision_stump(), stage.weak_classifier.level1_node);

        set_decision_stump(level2_true_node_p->decision_stump(), stage_weight, stage.weak_classifier.level2_true_node);
        set_decision_stump(level2_false_node_p->decision_stump(), stage_weight, stage.weak_classifier.level2_false_node);
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage with ?? weak classifiers, not yet implemented");
    }

    stage.weak_classifier.compute_bounding_box();
    return;
}

/// chooses which variant should be instanciated
SoftCascadeOverIntegralChannelsModel::variant_stages_t
create_empty_stages_vector(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data)
{
    SoftCascadeOverIntegralChannelsModel::variant_stages_t stages;

    typedef doppia_protobuf::SoftCascadeOverIntegralChannelsStage stage_data_t;

    const stage_data_t::FeatureTypes feature_type = stage_data.feature_type();


    if((feature_type == stage_data_t::Level2DecisionTree) and stage_data.has_level2_decision_tree())
    {
        //stages = stages_t(); // old stages style is now deprecated
        stages = fast_stages_t();
    }
    else
    {
        throw std::invalid_argument("SoftCascadeOverIntegralChannelsModel received an unknown stage type");
    }

    return stages;
}




class push_back_into_variant_stages
        : public boost::static_visitor<void>
{
public:

    typedef doppia_protobuf::SoftCascadeOverIntegralChannelsStage stage_data_t;

    const stage_data_t &stage_data;
    const int c;

    push_back_into_variant_stages(const stage_data_t &stage_data_, const int c_)
        : stage_data(stage_data_), c(c_)
    {
        // nothing to do here
        return;
    }

    template<typename T>
    void operator()(std::vector<T> &stages) const;
}; // end of visitor class push_back_into_variant_stages


template<typename T>
void push_back_into_variant_stages::operator()(std::vector<T> &stages) const
{
    printf("Received type %s\n", typeid(T).name());
    // fast_fractional_stages_t falls here
    throw std::invalid_argument("SoftCascadeOverIntegralChannelsModel "
                                "adding into a not implemented (deprecated?) stages type");
}

template<>
void push_back_into_variant_stages::operator()<fast_stage_t>(std::vector<fast_stage_t> &stages) const
{
    if(stage_data.has_level2_decision_tree() == false)
    {
        throw std::runtime_error("Stage data does not have the same type as first stage.");
    }

    fast_stage_t stage;

    //stage.weight = stage_data.weight();
    stage.cascade_threshold = stage_data.cascade_threshold();
    set_weak_classifier(stage_data, stage);

    //stages.push_back(stage); stage_t usage is now deprecated
    stages.push_back(stage);

    if(true and ((c == 0) or ((c +1) % 1000 == 0)) )
    {
        printf("Stage %i cascade_threshold == %.6f\n", c, stage.cascade_threshold);
    }
    return;
}

void SoftCascadeOverIntegralChannelsModel::set_stages_from_model(
        const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model)
{
    typedef google::protobuf::RepeatedPtrField< doppia_protobuf::IntegralChannelsFeature > features_t;

    log_info() << "The soft cascade contains " << model.stages().size() << " stages" << std::endl;


    // we select the stages type based on the first stage
    stages = create_empty_stages_vector(model.stages(0));

    for(int c=0; c < model.stages().size() ; c+=1)
    {
        const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data = model.stages(c);

        boost::apply_visitor(push_back_into_variant_stages(stage_data, c), stages);

    } // end of "for each stage in the cascade"

    return;
}

/// Helper method that gives the crucial information for the FPDW implementation
/// these numbers are obtained via
/// doppia/src/test/objects_detection/test_objects_detection + plot_channel_statistics.py
/// (this method is not speed critical)
float get_channel_scaling_factor(const boost::uint8_t channel_index,
                                 const float relative_scale)
{
    float channel_scaling = 1, up_a = 1, down_a = 1, up_b = 2, down_b = 2;

    // FIXME how to propagate here which method is used ?
    // should these values be part of the computing structure ?

    if(relative_scale == 1)
    { // when no rescaling there is no scaling factor
        return 1.0f;
    }

    const bool
            use_p_dollar_estimates = true,
            use_v0_estimates = false,
            use_no_estimate = false;

    if(use_p_dollar_estimates)
    {
        const float lambda = 1.099, a = 0.89;

        if(channel_index <= 6)
        { // Gradient histograms and gradient magnitude
            down_a = a; down_b = lambda / log(2);

            // upscaling case is roughly a linear growth
            // these are the ideal values
            up_a = 1; up_b = 0;
        }
        else if((channel_index >= 7) and (channel_index <= 9))
        { // LUV channels, quadratic growth
            // these are the ideal values
            down_a = 1; down_b = 2;
            up_a = 1; up_b = 2;
        }
        else
        {
            throw std::runtime_error("get_channel_scaling_factor use_p_dollar_estimates called with "
                                     "an unknown integral channel index");
        }

    }
    else if(use_v0_estimates)
    {
        // these values hold for IntegralChannelsForPedestrians::compute_v0
        // FIXME these are old values, need update

        // num_scales ==  12
        // r = a*(k**b); r: feature scaling factor; k: relative scale
        // HOG	for downscaling r = 0.989*(x**-1.022), for upscaling  r = 1.003*(x**1.372)
        // L	for downscaling r = 0.963*(x**-1.771), for upscaling  r = 0.956*(x**1.878)
        // UV	for downscaling r = 0.966*(x**-2.068), for upscaling  r = 0.962*(x**2.095)

        if(channel_index <= 6)
        { // Gradient histograms and gradient magnitude
            down_a = 1.0f/0.989; down_b = +1.022;
            // upscaling case is roughly a linear growth
            up_a = 1.003; up_b = 1.372;
        }
        else if(channel_index == 7)
        { // L channel, quadratic growth
            // for some strange reason test_objects_detection + plot_channel_statistics.py indicate
            // that the L channel behaves differently than UV channels
            down_a = 1.0f/0.963; down_b = +1.771;
            up_a = 0.956; up_b = 1.878;
        }
        else if(channel_index == 8 or channel_index ==9)
        { // UV channels, quadratic growth
            down_a = 1.0f/0.966; down_b = +2.068;
            up_a = 0.962; up_b = 2.095;
        }
        else
        {
            throw std::runtime_error("get_channel_scaling_factor use_v0_estimates called with "
                                     "an unknown integral channel index");
        }
    } // end of "IntegralChannlesForPedestrians::compute_v0"
    else if(use_no_estimate)
    {
        // we disregard the scaling and keep the same feature value
        up_a = 1; up_b = 0;
        down_a = 1; down_b = 0;
    }
    else
    {
        throw std::runtime_error("no estimate was selected for get_channel_scaling_factor");
    }


    {
        float a=1, b=2;
        if(relative_scale >= 1)
        { // upscaling case
            a = up_a;
            b = up_b;
        }
        else
        { // size_scaling < 1, downscaling case
            a = down_a;
            b = down_b;
        }

        channel_scaling = a*pow(relative_scale, b);

        const bool check_scaling = true;
        if(check_scaling)
        {
            if(relative_scale >= 1)
            { // upscaling
                if(channel_scaling < 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
            else
            { // downscaling
                if(channel_scaling > 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
        } // end of check_scaling
    }

    return channel_scaling;
}


void scale_the_box(IntegralChannelsFeature::rectangle_t &box, const float relative_scale)
{
    using boost::math::iround;
    box.min_corner().x( iround(box.min_corner().x() * relative_scale) );
    box.min_corner().y( iround(box.min_corner().y() * relative_scale) );
    box.max_corner().x( std::max(box.min_corner().x() + 1, iround(box.max_corner().x() * relative_scale)) );
    box.max_corner().y( std::max(box.min_corner().y() + 1, iround(box.max_corner().y() * relative_scale)) );

    assert(rectangle_area(box) >= 1);
    return;
}

/// we change the size of the rectangle and
/// adjust the threshold to take into the account the slight change in area
template<typename StumpType>
void scale_the_stump(StumpType &decision_stump,
                     const float relative_scale)
{

    const float channel_scaling_factor = get_channel_scaling_factor(
                decision_stump.feature.channel_index, relative_scale);

    typename StumpType::feature_t::rectangle_t &box = decision_stump.feature.box;
    const float original_area = rectangle_area(box);
    scale_the_box(box, relative_scale);
    const float new_area = rectangle_area(box);

    float area_approximation_scaling_factor = 1;
    if((new_area != 0) and (original_area != 0))
    {
        // integral_over_new_area * (original_area / new_area) =(approx)= integral_over_original_area
        //area_approximation_scaling_factor = original_area / new_area;
        const float expected_new_area = original_area*relative_scale*relative_scale;
        area_approximation_scaling_factor = expected_new_area / new_area;
        //printf("area_approximation_scaling_factor %.3f\n", area_approximation_scaling_factor);
    }

    decision_stump.feature_threshold /= area_approximation_scaling_factor; // FIXME this seems wrong !

    decision_stump.feature_threshold *= channel_scaling_factor;

    const bool print_channel_scaling_factor = false;
    if(print_channel_scaling_factor)
    {
        printf("relative_scale %.3f -> channel_scaling_factor %.3f\n", relative_scale, channel_scaling_factor);
    }

    return;
}


class get_rescaled_variant_stages
        : public boost::static_visitor<variant_stages_t>
{
public:

    const float relative_scale;

    get_rescaled_variant_stages(const float relative_scale_)
        : relative_scale(relative_scale_)
    {
        // nothing to do here
        return;
    }

    template<typename T>
    variant_stages_t operator()(const std::vector<T> &) const;

}; // end of visitor class get_rescaled_variant_stages


template<typename T>
variant_stages_t get_rescaled_variant_stages::operator()(const std::vector<T> &) const
{
    printf("Received type %s\n", typeid(T).name());
    throw std::invalid_argument("SoftCascadeOverIntegralChannelsModel "
                                "get_rescaled_stages received stage type not yet implemented (deprecated?)");

    return variant_stages_t(); // just because we need to return something
}

template<>
variant_stages_t get_rescaled_variant_stages::operator()<fast_stage_t>(const fast_stages_t &stages) const
{
    fast_stages_t rescaled_stages;
    rescaled_stages.reserve(stages.size());

    BOOST_FOREACH(const fast_stage_t &stage, stages)
    {
        fast_stage_t rescaled_stage = stage;

        fast_stage_t::weak_classifier_t &weak_classifier = rescaled_stage.weak_classifier;
        scale_the_stump(weak_classifier.level1_node, relative_scale);
        scale_the_stump(weak_classifier.level2_true_node, relative_scale);
        scale_the_stump(weak_classifier.level2_false_node, relative_scale);

        weak_classifier.compute_bounding_box(); // we update the bounding box of the weak classifier
        rescaled_stages.push_back(rescaled_stage);
    }

    if(false and (not stages.empty()))
    {
        printf("SoftCascadeOverIntegralChannelsModel::get_rescaled_fast_stages "
               "Rescaled stage 0 cascade_threshold == %3.f\n",
               rescaled_stages[0].cascade_threshold);
    }

    return rescaled_stages;
}

SoftCascadeOverIntegralChannelsModel::variant_stages_t
SoftCascadeOverIntegralChannelsModel::get_rescaled_stages(const float relative_scale) const
{
    return boost::apply_visitor(get_rescaled_variant_stages(relative_scale), stages);
}



SoftCascadeOverIntegralChannelsModel::variant_stages_t &
SoftCascadeOverIntegralChannelsModel::get_stages()
{
    return stages;
}


const SoftCascadeOverIntegralChannelsModel::variant_stages_t &
SoftCascadeOverIntegralChannelsModel::get_stages() const
{
    return stages;
}


int SoftCascadeOverIntegralChannelsModel::get_shrinking_factor() const
{
    return shrinking_factor;
}


class stages_are_empty: public boost::static_visitor<bool>
{
public:

    template<typename T>
    bool operator()(const T &stages) const
    {
        return stages.empty();
    }
}; // end of visitor class stages_are_empty


class get_last_cascade_threshold_from_variant_stages
        : public boost::static_visitor<float>
{
public:

    template<typename T>
    float operator()(const T &stages) const
    {
        if(stages.empty() == true)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::get_last_cascade_threshold " \
                                     "cascade classifier is empty");
        }
        const float last_cascade_threshold = stages.back().cascade_threshold;
        return last_cascade_threshold;
    }

    /*template<typename T>
    float operator()(const T &stages) const
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::get_last_cascade_threshold " \
                                 "failed to find the stages");
        return 0;
    }*/

}; // end of visitor class get_last_cascade_threshold_from_variant_stages


float SoftCascadeOverIntegralChannelsModel::get_last_cascade_threshold() const
{
    return boost::apply_visitor(get_last_cascade_threshold_from_variant_stages(), stages);
}

std::string SoftCascadeOverIntegralChannelsModel::get_semantic_category() const
{
    return semantic_category;
}
float SoftCascadeOverIntegralChannelsModel::get_scale() const
{
    return scale;
}


float SoftCascadeOverIntegralChannelsModel::get_occlusion_level() const
{
    return occlusion_level;
}


SoftCascadeOverIntegralChannelsModel::occlusion_type_t SoftCascadeOverIntegralChannelsModel::get_occlusion_type() const
{
    return occlusion_type;
}


std::string get_occlusion_type_name(const SoftCascadeOverIntegralChannelsModel::occlusion_type_t occlusion_type)
{
    std::string name = "unknown";
    switch(occlusion_type)
    {

    case SoftCascadeOverIntegralChannelsModel::LeftOcclusion:
        name = "left";
        break;

    case SoftCascadeOverIntegralChannelsModel::RightOcclusion:
        name = "right";
        break;

    case SoftCascadeOverIntegralChannelsModel::TopOcclusion:
        name = "top";
        break;

    case SoftCascadeOverIntegralChannelsModel::BottomOcclusion:
        name = "bottom";
        break;

    case SoftCascadeOverIntegralChannelsModel::NoOcclusion:
        name = "no occlusion";
        break;

    default:
        name = "unknown";
        break;
    }

    return name;
}


const SoftCascadeOverIntegralChannelsModel::model_window_size_t &SoftCascadeOverIntegralChannelsModel::get_model_window_size() const
{
    return model_window_size;
}


const SoftCascadeOverIntegralChannelsModel::object_window_t &SoftCascadeOverIntegralChannelsModel::get_object_window() const
{
    return object_window;
}


bool SoftCascadeOverIntegralChannelsModel::has_soft_cascade() const
{
    bool use_the_detector_model_cascade = false;

    const bool empty_cascade = boost::apply_visitor(stages_are_empty(), stages);
    if(empty_cascade == false)
    {
        // if the last weak learner has a "non infinity" cascade threshold,
        // then the model has a non trivial cascade, we should use it
        const float last_cascade_threshold = get_last_cascade_threshold();
        use_the_detector_model_cascade = (last_cascade_threshold > -1E5);
    }

    if(use_the_detector_model_cascade)
    {
        log_info() << "The detector model seems to include a cascade thresholds" << std::endl;
    }
    else
    {
        log_info() << "Will ignore the trivial cascade thresholds of the detector model" << std::endl;
    }

    return use_the_detector_model_cascade;
}

class shift_stages_visitor_t: public boost::static_visitor<void>
{
protected:
    const float x_shift, y_shift;
public:
    shift_stages_visitor_t(const float x_shift_, const float y_shift_);

    template<typename StageType>
    void operator()(std::vector<StageType> &stages) const;

protected:

    void shift_feature(IntegralChannelsFeature &feature) const;
};

shift_stages_visitor_t::shift_stages_visitor_t(const float x_shift_, const float y_shift_)
    : x_shift(x_shift_), y_shift(y_shift_)
{
    // nothing to do here
    return;
}

template<typename StageType>
void shift_stages_visitor_t::operator()(std::vector<StageType> &) const
{
    printf("Received type %s\n", typeid(StageType).name());
    throw std::invalid_argument("shift_stages_visitor_t received stage type not yet implemented (deprecated?)");

    return;
}

template<>
void shift_stages_visitor_t::operator()<fast_stage_t>(fast_stages_t &stages) const
{
    BOOST_FOREACH(fast_stage_t &stage, stages)
    {
        shift_feature(stage.weak_classifier.level1_node.feature);
        shift_feature(stage.weak_classifier.level2_true_node.feature);
        shift_feature(stage.weak_classifier.level2_false_node.feature);
    } // end of "for each stage"
    return;
}


void shift_stages_visitor_t::shift_feature(IntegralChannelsFeature &feature) const
{
    IntegralChannelsFeature::rectangle_t &box = feature.box;
    box.min_corner().x(std::max(0.0f, box.min_corner().x() + x_shift));
    box.min_corner().y(std::max(0.0f, box.min_corner().y() + y_shift));

    box.max_corner().x(std::max(0.0f, box.max_corner().x() + x_shift));
    box.max_corner().y(std::max(0.0f, box.max_corner().y() + y_shift));
    return;
}


void SoftCascadeOverIntegralChannelsModel::shift_stages_by_occlusion_level()
{
    float x_shift = 0, y_shift = 0;
    if(occlusion_type == LeftOcclusion)
    {
        x_shift = -model_window_size.x()*occlusion_level/shrinking_factor;
    }
    else if(occlusion_type == TopOcclusion)
    {
        y_shift = -model_window_size.y()*occlusion_level/shrinking_factor;
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::shift_stages_by_occlusion_level "
                                 "was called for a model with an occlusion type that does not need shifting");
    }

    log_info() << "Shifting model by (x,y) == (" << x_shift << ", " << y_shift << ") [shrunk pixels]" << std::endl;

    shift_stages_visitor_t visitor(x_shift, y_shift);
    boost::apply_visitor(visitor, stages);

    return;
}


/// Sanity check for the model, is its size/occlusion level/type consistent with its stages ?
void SoftCascadeOverIntegralChannelsModel::sanity_check() const
{

    log_debug() << boost::str(
                       boost::format("Checking model with occlusion '%s' %0.3f\n")
                       % get_occlusion_type_name(occlusion_type)
                       % occlusion_level);

    DetectorSearchRange dummy_search_range;
    dummy_search_range.detection_window_scale = 1.0;
    dummy_search_range.detection_window_ratio = 1.0;
    dummy_search_range.range_scaling = 1.0;
    dummy_search_range.range_ratio = 1.0;
    dummy_search_range.detector_occlusion_level = occlusion_level;
    dummy_search_range.detector_occlusion_type = occlusion_type;
    dummy_search_range.min_x = 0;
    dummy_search_range.max_x = 0;
    dummy_search_range.min_y = 0;
    dummy_search_range.max_y = 0;

    const float image_to_channel = 1.0f/shrinking_factor;

    int shrunk_width = 0, shrunk_height = 0;

    if(occlusion_type == SoftCascadeOverIntegralChannelsModel::NoOcclusion)
    {
        shrunk_width = std::ceil(model_window_size.x() * image_to_channel);
        shrunk_height = std::ceil(model_window_size.y() * image_to_channel);
    }
    else if (occlusion_type == SoftCascadeOverIntegralChannelsModel::LeftOcclusion)
    {
        // the sanity check is done after shifting the model,
        // so we can do the same check as with Right occlusion
        //dummy_search_range.min_x = std::max<int>(0, std::floor(model_window_size.x() * image_to_channel * occlusion_level) - 1);
        //shrunk_width = std::ceil(model_window_size.x() * image_to_channel);

        shrunk_width = std::ceil(model_window_size.x() * image_to_channel * (1 - occlusion_level));
        shrunk_height = std::ceil(model_window_size.y() * image_to_channel);
    }
    else if (occlusion_type == SoftCascadeOverIntegralChannelsModel::RightOcclusion)
    {
        shrunk_width = std::ceil(model_window_size.x() * image_to_channel * (1 - occlusion_level));
        shrunk_height = std::ceil(model_window_size.y() * image_to_channel);
    }
    else if (occlusion_type == SoftCascadeOverIntegralChannelsModel::BottomOcclusion)
    {
        shrunk_width = std::ceil(model_window_size.x() * image_to_channel);
        shrunk_height = std::ceil(model_window_size.y() * image_to_channel * (1 - occlusion_level));
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::sanity_check "
                                 "received an unhandled occlusion type");
    }

    const bool must_check_borders = false; // FIXME a better version would use true
    const bool should_touch_borders = must_check_borders and (occlusion_level <= 0); // if no occlusion, then should do also borders check
    check_stages_and_range_visitor visitor(
                -1,
                dummy_search_range,
                shrunk_width, shrunk_height,
                should_touch_borders);
    bool everything_is_fine = boost::apply_visitor(visitor, stages);

    if(not everything_is_fine)
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::sanity_check failed");
    }
    else
    {
        printf("Single scale/occlusion model sanity check passed.\n");
    }

    return;
}


} // end of namespace doppia
