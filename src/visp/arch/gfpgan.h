#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::gfpgan {

tensor modulated_conv(model_ref m, tensor x, tensor latent, bool demodulate);
tensor style_conv(model_ref m, tensor x, tensor latent, bool upsample);

} // namespace visp::gfpgan
