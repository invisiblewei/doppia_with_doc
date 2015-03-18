#include "GpuIntegralChannelsModelsBundleDetector.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"

#include <boost/foreach.hpp>

#include <stdexcept>

namespace doppia {

MODULE_LOG_MACRO("GpuIntegralChannelsModelsBundleDetector")

GpuIntegralChannelsModelsBundleDetector::GpuIntegralChannelsModelsBundleDetector(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold)
    : BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   non_maximal_suppression_p, score_threshold),
      GpuIntegralChannelsDetector(
          options,
          boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
          non_maximal_suppression_p,
          score_threshold),
      BaseIntegralChannelsModelsBundleDetector(options,
                                               detector_model_p)
{
}


GpuIntegralChannelsModelsBundleDetector::~GpuIntegralChannelsModelsBundleDetector()
{
    // nothing to do here
    return;
}


} // end of namespace doppia

