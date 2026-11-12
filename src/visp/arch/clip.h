#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::clip {

tensor embeddings(model_ref m, tensor x, clip_params const& p);
tensor mlp(model_ref m, tensor x, clip_params const& p);
tensor self_attention(model_ref m, tensor x, clip_params const& p);
tensor layer(model_ref m, tensor x, clip_params const& p);
clip_output encode(model_ref m, tensor image, clip_params const& p);

} // namespace visp::clip
