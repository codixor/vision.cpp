#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::yolox {

tensor base_conv(model_ref m, tensor x, int stride);
tensor csp_layer(model_ref m, tensor x, int n, bool shortcut);
tensor spp(model_ref m, tensor x);

} // namespace visp::yolox
