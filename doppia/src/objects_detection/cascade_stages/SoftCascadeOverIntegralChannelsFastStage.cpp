#include "SoftCascadeOverIntegralChannelsFastStage.hpp"

#include <algorithm> // for min and max

namespace doppia {

DecisionStumpWithWeights::DecisionStumpWithWeights(const DecisionStump &stump, const float feature_weight)
{

    feature = stump.feature;
    feature_threshold = stump.feature_threshold;

    if(stump.larger_than_threshold)
    {
        weight_true_leaf = feature_weight;
        weight_false_leaf = -feature_weight;
    }
    else
    {
        weight_true_leaf = -feature_weight;
        weight_false_leaf = feature_weight;
    }
    return;
}

void Level2DecisionTreeWithWeights::compute_bounding_box()
{
    bounding_box = level1_node.feature.box;

    rectangle_t &bb = bounding_box;
    const rectangle_t
            &bb_a = level2_true_node.feature.box,
            &bb_b = level2_false_node.feature.box;

    bb.min_corner().x( std::min(bb.min_corner().x(), bb_a.min_corner().x()) );
    bb.min_corner().x( std::min(bb.min_corner().x(), bb_b.min_corner().x()) );

    bb.min_corner().y( std::min(bb.min_corner().y(), bb_a.min_corner().y()) );
    bb.min_corner().y( std::min(bb.min_corner().y(), bb_b.min_corner().y()) );

    bb.max_corner().x( std::max(bb.max_corner().x(), bb_a.max_corner().x()) );
    bb.max_corner().x( std::max(bb.max_corner().x(), bb_b.max_corner().x()) );

    bb.max_corner().y( std::max(bb.max_corner().y(), bb_a.max_corner().y()) );
    bb.max_corner().y( std::max(bb.max_corner().y(), bb_b.max_corner().y()) );

    return;
}

/*
DecisionStumpWithWeights::~DecisionStumpWithWeights()
{
    // nothing to do here
    return;
}*/


} // end of namespace doppia
