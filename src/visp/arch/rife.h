#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::rife {

tensor warp(model_ref m, tensor img, tensor flow, int flow_channel);
tensor if_block(model_ref m, tensor x, tensor flow, int scale);

} // namespace visp::rife
