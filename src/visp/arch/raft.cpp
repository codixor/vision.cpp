// RAFT optical flow (Teed & Deng; torchvision weights, BSD-3-Clause). raft_large only.
// Feature/context encoders at 1/8 resolution, all-pairs correlation pyramid, ConvGRU
// refinement steps and learned convex 8x upsampling. Runs in WHCN layout.
// The correlation lookup (grid_sample, zero padding) is built from per-pixel get_rows gathers.
// Weights: raft_large_*.pth -> pickle2safetensors -> visp-convert raft (batch norms fused).
// Input size must be a multiple of 8 (the caller pads).

#include "visp/arch/raft.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <array>
#include <cmath>
#include <string_view>

namespace visp {
namespace raft {

tensor conv(model_ref m, tensor x, int stride) {
    tensor w = m.weights("weight");
    int kw = int(w->ne[0]), kh = int(w->ne[1]);
    x = ggml_conv_2d_direct(m, w, x, stride, stride, kw / 2, kh / 2, 1, 1);
    if (tensor b = m.find("bias")) {
        x = ggml_add(m, x, ggml_reshape_4d(m, b, 1, 1, b->ne[0], 1));
    }
    return x;
}

tensor channels(model_ref m, tensor x, int64_t first, int64_t count) {
    tensor v = ggml_view_4d(
        m, x, x->ne[0], x->ne[1], count, x->ne[3], x->nb[1], x->nb[2], x->nb[3], first * x->nb[2]);
    return ggml_cont(m, v);
}

tensor instance_norm(model_ref m, tensor x) {
    int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2], n = x->ne[3];
    x = ggml_norm(m, ggml_reshape_3d(m, ggml_cont(m, x), w * h, c, n), 1e-5f);
    return ggml_reshape_4d(m, x, w, h, c, n);
}

// Conv2dNormActivation: conv (+ instance norm in the feature encoder; batch norms are fused) + relu
tensor conv_norm(model_ref m, tensor x, int stride, bool inorm, bool relu = true) {
    x = conv(m[0], x, stride);
    if (inorm) {
        x = instance_norm(m, x);
    }
    return relu ? ggml_relu(m, x) : x;
}

tensor residual_block(model_ref m, tensor x, int stride, bool inorm) {
    tensor y = conv_norm(m["convnormrelu1"], x, stride, inorm);
    y = conv_norm(m["convnormrelu2"], y, 1, inorm);
    if (m["downsample"][0].find("weight")) {
        x = conv_norm(m["downsample"], x, stride, inorm, false);
    }
    return ggml_relu(m, ggml_add(m, x, y));
}

tensor encoder(model_ref m, tensor x, bool inorm) {
    x = conv_norm(m["convnormrelu"], x, 2, inorm);
    int const strides[] = {1, 2, 2};
    char const* const layers[] = {"layer1", "layer2", "layer3"};
    for (int i = 0; i < 3; ++i) {
        x = residual_block(m[layers[i]][0], x, strides[i], inorm);
        x = residual_block(m[layers[i]][1], x, 1, inorm);
    }
    return conv(m["conv"], x, 1);
}

// Bilinear lookup of each pixel's (2r+1)^2 neighbourhood in every pyramid level.
// cx, cy: [1, P] centroid coordinates at level 0. Returns WHCN [w, h, levels*(2r+1)^2].
tensor lookup(model_ref m, span<tensor const> pyramid, tensor cx, tensor cy, int r, int64_t w,
              int64_t h) {
    int64_t const n = 2 * r + 1, k = n * n, p = w * h;
    tensor off = ggml_arange(m, float(-r), float(r + 1), 1.f);
    // torch meshgrid(indexing="ij") stacked as (x, y): the outer index offsets x
    tensor dy = ggml_repeat_4d(m, ggml_reshape_2d(m, off, n, 1), n, n, 1, 1);
    tensor dx = ggml_repeat_4d(m, ggml_reshape_2d(m, off, 1, n), n, n, 1, 1);
    dx = ggml_repeat_4d(m, ggml_reshape_2d(m, dx, k, 1), k, p, 1, 1);
    dy = ggml_repeat_4d(m, ggml_reshape_2d(m, dy, k, 1), k, p, 1, 1);

    tensor features = nullptr;
    for (size_t l = 0; l < pyramid.size(); ++l) {
        tensor corr = pyramid[l];
        int64_t const wl = corr->ne[0], hl = corr->ne[1];
        float const s = 1.f / float(1 << l);
        tensor x = ggml_add(m, dx, ggml_scale(m, cx, s));
        tensor y = ggml_add(m, dy, ggml_scale(m, cy, s));
        tensor x0 = ggml_floor(m, x);
        tensor y0 = ggml_floor(m, y);
        tensor wx = ggml_sub(m, x, x0);
        tensor wy = ggml_sub(m, y, y0);
        tensor x1 = ggml_scale_bias(m, x0, 1.f, 1.f);
        tensor y1 = ggml_scale_bias(m, y0, 1.f, 1.f);
        tensor src = ggml_reshape_3d(m, corr, 1, wl * hl, p);

        // weight of an integer coordinate: 1 inside [0, size-1], 0 outside (zero padding)
        auto inside = [&](tensor v, int64_t size) {
            tensor lo = ggml_clamp(m, ggml_scale_bias(m, v, 1.f, 1.f), 0.f, 1.f);
            tensor hi = ggml_clamp(m, ggml_scale_bias(m, v, -1.f, float(size)), 0.f, 1.f);
            return ggml_mul(m, lo, hi);
        };
        auto corner = [&](tensor xi, tensor yi, tensor weight) {
            // ggml_clamp works in place: clamp copies so xi/yi stay intact for the weights
            tensor xc = ggml_clamp(m, ggml_dup(m, xi), 0.f, float(wl - 1));
            tensor yc = ggml_clamp(m, ggml_dup(m, yi), 0.f, float(hl - 1));
            tensor idx = ggml_add(m, ggml_scale(m, yc, float(wl)), xc);
            idx = ggml_cast(m, idx, GGML_TYPE_I32);
            tensor v = ggml_reshape_2d(m, ggml_get_rows(m, src, idx), k, p);
            weight = ggml_mul(m, weight, ggml_mul(m, inside(xi, wl), inside(yi, hl)));
            return ggml_mul(m, v, weight);
        };
        tensor one_wx = ggml_scale_bias(m, wx, -1.f, 1.f);
        tensor one_wy = ggml_scale_bias(m, wy, -1.f, 1.f);
        tensor v = corner(x0, y0, ggml_mul(m, one_wx, one_wy));
        v = ggml_add(m, v, corner(x1, y0, ggml_mul(m, wx, one_wy)));
        v = ggml_add(m, v, corner(x0, y1, ggml_mul(m, one_wx, wy)));
        v = ggml_add(m, v, corner(x1, y1, ggml_mul(m, wx, wy)));
        features = features ? ggml_concat(m, features, v, 0) : v;
    }
    tensor t = ggml_cont(m, ggml_transpose(m, features)); // [P, levels*k]
    return ggml_reshape_4d(m, t, w, h, t->ne[1], 1);
}

tensor gru(model_ref m, tensor h, tensor x) {
    tensor hx = ggml_concat(m, h, x, 2);
    tensor z = ggml_sigmoid(m, conv(m["convz"], hx, 1));
    tensor r = ggml_sigmoid(m, conv(m["convr"], hx, 1));
    tensor q = ggml_tanh(m, conv(m["convq"], ggml_concat(m, ggml_mul(m, r, h), x, 2), 1));
    return ggml_add(m, h, ggml_mul(m, z, ggml_sub(m, q, h)));
}

// Convex 8x upsampling: each output pixel is a softmax-weighted mix of the 3x3 coarse neighbours.
tensor upsample(model_ref m, tensor flow, tensor mask) {
    int64_t const w = flow->ne[0], h = flow->ne[1], p = w * h;
    // zero border of one pixel (built by concat: Metal has no PAD with left padding)
    tensor padded = ggml_scale(m, flow, 8.f);
    tensor zx = ggml_scale(m, ggml_cont(m, ggml_view_4d(m, padded, 1, h, 2, 1, padded->nb[1],
                                                        padded->nb[2], padded->nb[3], 0)), 0.f);
    padded = ggml_concat(m, ggml_concat(m, zx, padded, 0), zx, 0);
    tensor zy = ggml_scale(m, ggml_cont(m, ggml_view_4d(m, padded, w + 2, 1, 2, 1, padded->nb[1],
                                                        padded->nb[2], padded->nb[3], 0)), 0.f);
    padded = ggml_concat(m, ggml_concat(m, zy, padded, 1), zy, 1);
    tensor unfolded = nullptr;
    for (int ki = 0; ki < 3; ++ki) {
        for (int kj = 0; kj < 3; ++kj) {
            tensor v = ggml_view_4d(m, padded, w, h, 2, 1, padded->nb[1], padded->nb[2],
                                    padded->nb[3], kj * padded->nb[0] + ki * padded->nb[1]);
            v = ggml_reshape_3d(m, ggml_cont(m, v), 1, p, 2);
            unfolded = unfolded ? ggml_concat(m, unfolded, v, 0) : v; // [9, P, 2]
        }
    }
    tensor s = ggml_reshape_3d(m, ggml_cont(m, mask), p, 64, 9);
    s = ggml_soft_max(m, ggml_cont(m, ggml_permute(m, s, 1, 2, 0, 3))); // [9, P, 64]
    tensor out = nullptr;
    for (int c = 0; c < 2; ++c) {
        tensor u = ggml_view_3d(m, unfolded, 9, p, 1, unfolded->nb[1], unfolded->nb[2],
                                c * unfolded->nb[2]);
        tensor r = ggml_sum_rows(m, ggml_mul(m, s, ggml_cont(m, u))); // [1, P, 64]
        r = ggml_reshape_4d(m, r, w, h, 8, 8);                       // [x, y, b, a]
        r = ggml_cont(m, ggml_permute(m, r, 1, 3, 0, 2));             // [b, x, a, y]
        r = ggml_reshape_4d(m, r, 8 * w, 8 * h, 1, 1);
        out = out ? ggml_concat(m, out, r, 2) : r;
    }
    return out;
}

} // namespace raft

tensor raft_flow(model_ref m, tensor image0, tensor image1, raft_params const& p) {
    using namespace raft;
    tensor img0 = ggml_scale_bias(m, cwhn_to_contiguous_2d(m, image0), 2.f, -1.f);
    tensor img1 = ggml_scale_bias(m, cwhn_to_contiguous_2d(m, image1), 2.f, -1.f);
    tensor f0 = encoder(m["feature_encoder"], img0, true);
    tensor f1 = encoder(m["feature_encoder"], img1, true);
    int64_t const w = f0->ne[0], h = f0->ne[1], c = f0->ne[2], n = w * h;

    // all-pairs correlation: row p0 is a [w, h] map over frame-1 pixels
    tensor t0 = ggml_cont(m, ggml_transpose(m, ggml_reshape_2d(m, f0, n, c))); // [c, P]
    tensor t1 = ggml_cont(m, ggml_transpose(m, ggml_reshape_2d(m, f1, n, c)));
    tensor corr = ggml_scale(m, ggml_mul_mat(m, t1, t0), 1.f / std::sqrt(float(c)));
    std::array<tensor, 8> pyramid{};
    pyramid[0] = ggml_reshape_3d(m, corr, w, h, n);
    for (int l = 1; l < p.corr_levels; ++l) {
        pyramid[l] = ggml_pool_2d(m, pyramid[l - 1], GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    }
    span<tensor const> levels(pyramid.data(), size_t(p.corr_levels));

    tensor ctx = encoder(m["context_encoder"], img0, false);
    tensor hidden = ggml_tanh(m, channels(m, ctx, 0, p.hidden_size));
    tensor context = ggml_relu(m, channels(m, ctx, p.hidden_size, ctx->ne[2] - p.hidden_size));

    tensor gx = ggml_repeat_4d(m, ggml_arange(m, 0.f, float(w), 1.f), w, h, 1, 1);
    tensor gy = ggml_arange(m, 0.f, float(h), 1.f);
    gy = ggml_repeat_4d(m, ggml_reshape_2d(m, gy, 1, h), w, h, 1, 1);
    tensor flow = ggml_scale(m, channels(m, ctx, 0, 2), 0.f);

    model_ref ub = m["update_block"];
    model_ref me = ub["motion_encoder"];
    model_ref rb = ub["recurrent_block"];
    model_ref fh = ub["flow_head"];
    for (int it = 0; it < p.iterations; ++it) {
        tensor fx = ggml_reshape_2d(m, channels(m, flow, 0, 1), 1, n);
        tensor fy = ggml_reshape_2d(m, channels(m, flow, 1, 1), 1, n);
        tensor cx = ggml_add(m, ggml_reshape_2d(m, gx, 1, n), fx);
        tensor cy = ggml_add(m, ggml_reshape_2d(m, gy, 1, n), fy);
        tensor cf = lookup(m, levels, cx, cy, p.corr_radius, w, h);

        tensor cr = ggml_relu(m, conv(me["convcorr1"][0], cf, 1));
        cr = ggml_relu(m, conv(me["convcorr2"][0], cr, 1));
        tensor fl = ggml_relu(m, conv(me["convflow1"][0], flow, 1));
        fl = ggml_relu(m, conv(me["convflow2"][0], fl, 1));
        tensor motion = ggml_relu(m, conv(me["conv"][0], ggml_concat(m, cr, fl, 2), 1));
        motion = ggml_concat(m, motion, flow, 2);
        tensor x = ggml_concat(m, context, motion, 2);

        hidden = gru(rb["convgru1"], hidden, x);
        hidden = gru(rb["convgru2"], hidden, x);
        tensor delta = conv(fh["conv2"], ggml_relu(m, conv(fh["conv1"], hidden, 1)), 1);
        flow = ggml_add(m, flow, delta);
    }
    model_ref mp = m["mask_predictor"];
    tensor mask = ggml_relu(m, conv(mp["convrelu"][0], hidden, 1));
    mask = ggml_scale(m, conv(mp["conv"], mask, 1), 0.25f);
    tensor out = upsample(m, flow, mask);
    return compute_graph_output(m, out, "raft_flow");
}

raft_params raft_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "raft") {
        throw except("Architecture expected to be 'raft', but was '{}' ({})", arch, f.path);
    }
    raft_params p;
    p.hidden_size = f.get_int("raft.hidden_size");
    p.corr_levels = f.get_int("raft.corr_levels");
    p.corr_radius = f.get_int("raft.corr_radius");
    if (p.corr_levels < 1 || p.corr_levels > 8 || p.corr_radius < 1 || p.corr_radius > 8) {
        throw except("RAFT: unsupported pyramid {} levels / radius {}", p.corr_levels, p.corr_radius);
    }
    if (gguf_find_tensor(f.gguf.get(), "mask_predictor.conv.weight") < 0) {
        throw except("RAFT: only raft_large (with mask predictor) is supported ({})", f.path);
    }
    return p;
}

} // namespace visp
