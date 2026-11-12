#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::raft {

tensor encoder(model_ref m, tensor x, bool instance_norm);
tensor upsample(model_ref m, tensor flow, tensor mask);

} // namespace visp::raft
