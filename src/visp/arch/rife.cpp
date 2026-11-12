// RIFE (Real-Time Intermediate Flow Estimation) frame interpolation, Practical-RIFE IFNet v4.2x
// (hzwer, MIT). Coarse-to-fine flow blocks at 1/16 .. 1/1 scale with a small feature encoder.
// Runs in WHCN layout. grid_sample (bilinear, border, align_corners) is built from gathers.
// Weights: flownet.pkl -> pickle2safetensors -> visp-convert rife.
// Input size must be a multiple of 64 (the caller pads).

#include "visp/arch/rife.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <initializer_list>
#include <string_view>

namespace visp {
namespace rife {

tensor lrelu(model_ref m, tensor x) {
    return ggml_leaky_relu(m, x, 0.2f, true);
}

tensor conv(model_ref m, tensor x, int stride = 1) {
    int k = int(m.weights("weight")->ne[0]);
    return conv_2d(m, x, stride, k / 2);
}

tensor channels(model_ref m, tensor x, int64_t first, int64_t count) {
    tensor v = ggml_view_4d(
        m, x, x->ne[0], x->ne[1], count, x->ne[3], x->nb[1], x->nb[2], x->nb[3], first * x->nb[2]);
    return ggml_cont(m, v);
}

tensor resize(model_ref m, tensor x, int64_t w, int64_t h) {
    if (x->ne[0] == w && x->ne[1] == h) {
        return x;
    }
    return ggml_interpolate(m, x, w, h, x->ne[2], x->ne[3], GGML_SCALE_MODE_BILINEAR);
}

// ConvTranspose2d(k=4, stride=2, padding=1): full transposed conv, then crop one pixel per side.
tensor conv_transpose_p1(model_ref m, tensor x) {
    x = conv_transpose_2d(m, x, 2);
    int64_t w = x->ne[0] - 2, h = x->ne[1] - 2;
    x = ggml_view_4d(
        m, x, w, h, x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], x->nb[0] + x->nb[1]);
    return ggml_cont(m, x);
}

// PixelShuffle(2) for WHCN [W, H, 4C, 1] -> [2W, 2H, C, 1]
tensor pixel_shuffle2(model_ref m, tensor x) {
    int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2] / 4;
    x = ggml_reshape_4d(m, x, w, h, 2, 2 * c);       // [W, H, j, (i, c)]
    x = ggml_cont(m, ggml_permute(m, x, 1, 2, 0, 3)); // [j, W, H, (i, c)]
    x = ggml_reshape_4d(m, x, 2 * w, h, 2, c);        // [2W, H, i, c]
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3)); // [2W, i, H, c]
    return ggml_reshape_4d(m, x, 2 * w, 2 * h, c, 1);
}

// Backward warp: out(x, y) = img(x + fx, y + fy), bilinear with border clamp
// (F.grid_sample with align_corners=True and padding_mode="border").
tensor warp(model_ref m, tensor img, tensor flow, int flow_channel) {
    int64_t const w = img->ne[0], h = img->ne[1], c = img->ne[2];
    tensor fx = ggml_reshape_2d(m, channels(m, flow, flow_channel, 1), w, h);
    tensor fy = ggml_reshape_2d(m, channels(m, flow, flow_channel + 1, 1), w, h);
    tensor gx = ggml_repeat(m, ggml_reshape_2d(m, ggml_arange(m, 0.f, float(w), 1.f), w, 1), fx);
    tensor gy = ggml_repeat(m, ggml_reshape_2d(m, ggml_arange(m, 0.f, float(h), 1.f), 1, h), fy);
    tensor x = ggml_clamp(m, ggml_add(m, gx, fx), 0.f, float(w - 1));
    tensor y = ggml_clamp(m, ggml_add(m, gy, fy), 0.f, float(h - 1));
    tensor x0 = ggml_floor(m, x);
    tensor y0 = ggml_floor(m, y);
    tensor wx = ggml_sub(m, x, x0);
    tensor wy = ggml_sub(m, y, y0);
    tensor x1 = ggml_clamp(m, ggml_scale_bias(m, x0, 1.f, 1.f), 0.f, float(w - 1));
    tensor y1 = ggml_clamp(m, ggml_scale_bias(m, y0, 1.f, 1.f), 0.f, float(h - 1));

    tensor src = ggml_cont(m, permute_whcn_to_cwhn(m, img));
    src = ggml_reshape_2d(m, src, c, w * h);
    auto gather = [&](tensor xi, tensor yi) {
        tensor idx = ggml_add(m, ggml_scale(m, yi, float(w)), xi);
        idx = ggml_cast(m, ggml_reshape_1d(m, idx, w * h), GGML_TYPE_I32);
        return ggml_get_rows(m, src, idx); // [c, w*h]
    };
    auto weight = [&](tensor a) { return ggml_reshape_2d(m, a, 1, w * h); };
    tensor one_wx = ggml_scale_bias(m, wx, -1.f, 1.f);
    tensor one_wy = ggml_scale_bias(m, wy, -1.f, 1.f);
    tensor r = ggml_mul(m, gather(x0, y0), weight(ggml_mul(m, one_wx, one_wy)));
    r = ggml_add(m, r, ggml_mul(m, gather(x1, y0), weight(ggml_mul(m, wx, one_wy))));
    r = ggml_add(m, r, ggml_mul(m, gather(x0, y1), weight(ggml_mul(m, one_wx, wy))));
    r = ggml_add(m, r, ggml_mul(m, gather(x1, y1), weight(ggml_mul(m, wx, wy))));
    r = ggml_reshape_4d(m, r, c, w, h, 1);
    return ggml_cont(m, permute_cwhn_to_whcn(m, r));
}

tensor encode(model_ref m, tensor x) {
    x = lrelu(m, conv(m["cnn0"], x, 2));
    x = lrelu(m, conv(m["cnn1"], x));
    x = lrelu(m, conv(m["cnn2"], x));
    return conv_transpose_p1(m["cnn3"], x);
}

// Returns [W, H, 13, 1]: flow (4), mask (1), features (8) at full resolution.
tensor if_block(model_ref m, tensor x, tensor flow, int scale) {
    int64_t const w = x->ne[0], h = x->ne[1];
    x = resize(m, x, w / scale, h / scale);
    if (flow) {
        tensor f = ggml_scale(m, resize(m, flow, w / scale, h / scale), 1.f / float(scale));
        x = ggml_concat(m, x, f, 2);
    }
    tensor feat = lrelu(m, conv(m["conv0"][0][0], x, 2));
    feat = lrelu(m, conv(m["conv0"][1][0], feat, 2));
    model_ref blocks = m["convblock"];
    for (int i = 0; i < 8; ++i) {
        model_ref r = blocks[i];
        tensor beta = ggml_reshape_4d(m, r.weights("beta"), 1, 1, feat->ne[2], 1);
        tensor y = ggml_mul(m, conv(r["conv"], feat), beta);
        feat = lrelu(m, ggml_add(m, y, feat));
    }
    tensor tmp = pixel_shuffle2(m, conv_transpose_p1(m["lastconv"][0], feat));
    tmp = resize(m, tmp, w, h);
    tensor fl = ggml_scale(m, channels(m, tmp, 0, 4), float(scale));
    return ggml_concat(m, fl, channels(m, tmp, 4, tmp->ne[2] - 4), 2);
}

} // namespace rife

tensor rife_interpolate(model_ref m, tensor image0, tensor image1, float t, rife_params const& p) {
    using namespace rife;
    tensor img0 = ggml_clamp(m, cwhn_to_contiguous_2d(m, image0), 0.f, 1.f);
    tensor img1 = ggml_clamp(m, cwhn_to_contiguous_2d(m, image1), 0.f, 1.f);
    tensor f0 = encode(m["encode"], img0);
    tensor f1 = encode(m["encode"], img1);
    tensor timestep = ggml_scale_bias(m, channels(m, img0, 0, 1), 0.f, t);

    auto cat = [&](std::initializer_list<tensor> xs) {
        tensor r = nullptr;
        for (tensor x : xs) {
            r = r ? ggml_concat(m, r, x, 2) : x;
        }
        return r;
    };
    tensor flow = nullptr, mask = nullptr, feat = nullptr;
    tensor warped0 = img0, warped1 = img1;
    for (int i = 0; i < p.num_blocks; ++i) {
        int scale = 1 << (p.num_blocks - 1 - i); // 16, 8, 4, 2, 1
        model_ref block = m[format<fixed_string<16>>("block{}", i).c_str()];
        tensor out;
        if (!flow) {
            out = if_block(block, cat({img0, img1, f0, f1, timestep}), nullptr, scale);
            flow = channels(m, out, 0, 4);
        } else {
            tensor wf0 = warp(m, f0, flow, 0);
            tensor wf1 = warp(m, f1, flow, 2);
            tensor in = cat({warped0, warped1, wf0, wf1, timestep, mask, feat});
            out = if_block(block, in, flow, scale);
            flow = ggml_add(m, flow, channels(m, out, 0, 4));
        }
        mask = channels(m, out, 4, 1);
        feat = channels(m, out, 5, out->ne[2] - 5);
        warped0 = warp(m, img0, flow, 0);
        warped1 = warp(m, img1, flow, 2);
    }
    mask = ggml_sigmoid(m, mask);
    tensor inv = ggml_scale_bias(m, mask, -1.f, 1.f);
    tensor r = ggml_add(m, ggml_mul(m, warped0, mask), ggml_mul(m, warped1, inv));
    r = ggml_clamp(m, r, 0.f, 1.f);
    return compute_graph_output(m, contiguous_2d_to_cwhn(m, r), "rife_result");
}

rife_params rife_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "rife") {
        throw except("Architecture expected to be 'rife', but was '{}' ({})", arch, f.path);
    }
    rife_params p;
    p.num_blocks = f.get_int("rife.num_blocks");
    p.feature_channels = f.get_int("rife.feature_channels");
    if (p.num_blocks < 1 || p.num_blocks > 6) {
        throw except("RIFE: unsupported block count {}", p.num_blocks);
    }
    return p;
}

} // namespace visp
