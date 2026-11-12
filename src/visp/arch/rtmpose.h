#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::rtmpose {

tensor conv_module(model_ref m, tensor x, int stride = 1);
tensor cspnext_block(model_ref m, tensor x, bool identity);
tensor csp_layer(model_ref m, tensor x, int n, bool identity);
tensor rtmcc_head(model_ref m, tensor feat, rtmpose_params const& p);

} // namespace visp::rtmpose
