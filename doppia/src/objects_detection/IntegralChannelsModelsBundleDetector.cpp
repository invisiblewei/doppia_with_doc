#include "IntegralChannelsModelsBundleDetector.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>

#include <stdexcept>

namespace doppia {

MODULE_LOG_MACRO("IntegralChannelsModelsBundleDetector")

IntegralChannelsModelsBundleDetector::IntegralChannelsModelsBundleDetector(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold)
    : BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   non_maximal_suppression_p, score_threshold),
      IntegralChannelsDetector(
          options,
          boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
          non_maximal_suppression_p,
          score_threshold),
      BaseIntegralChannelsModelsBundleDetector(options,
                                               detector_model_p)
{
    return;
}


IntegralChannelsModelsBundleDetector::~IntegralChannelsModelsBundleDetector()
{
    // nothing to do here
    return;
}


} // end of namespace doppia
