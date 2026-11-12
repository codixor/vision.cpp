// RTMPose / DWPose whole-body pose estimation (mmpose / DWPose, Apache-2.0).
// CSPNeXt backbone + RTMCC SimCC head. Weights: official .pth -> pickle2safetensors
// --key state_dict -> visp-convert rtmpose. Batch norm is fused into the convolutions.

#include "visp/arch/rtmpose.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace visp {
namespace rtmpose {

int channel_dim(model_ref const& m) {
    return is_cwhn(m) ? 0 : 2;
}

int kernel_size(model_ref const& m, tensor w) {
    return is_cwhn(m) ? int(w->ne[1]) : int(w->ne[0]);
}

// Conv + (fused BN) + SiLU, "same" padding.
tensor conv_module(model_ref m, tensor x, int stride) {
    int k = kernel_size(m, m["conv"].weights("weight"));
    x = conv_2d(m["conv"], x, stride, (k - 1) / 2);
    return ggml_silu_inplace(m, x);
}

tensor depthwise_module(model_ref m, tensor x) {
    tensor w = m["conv"].weights("weight");
    // CWHN depthwise weights are [c, 1, kw, kh]; WHCN are [kw, kh, 1, c].
    int k = is_cwhn(m) ? int(w->ne[2]) : int(w->ne[0]);
    x = conv_2d_depthwise(m["conv"], x, 1, (k - 1) / 2);
    return ggml_silu_inplace(m, x);
}

tensor cspnext_block(model_ref m, tensor x, bool identity) {
    tensor y = conv_module(m["conv1"], x);
    y = depthwise_module(m["conv2"]["depthwise_conv"], y);
    y = conv_module(m["conv2"]["pointwise_conv"], y);
    return identity ? ggml_add(m, y, x) : y;
}

// x * hardsigmoid(fc(mean_hw(x)))
tensor channel_attention(model_ref m, tensor x) {
    tensor pooled;
    if (is_cwhn(m)) { // [C, W, H, N] -> mean over W*H
        auto [c, w, h, n] = nelements(x);
        tensor t = ggml_reshape_3d(m, x, c, w * h, n);
        t = ggml_cont(m, ggml_permute(m, t, 1, 0, 2, 3)); // [WH, C, N]
        t = ggml_mean(m, t);                              // [1, C, N]
        pooled = ggml_reshape_4d(m, t, c, 1, 1, n);
    } else { // [W, H, C, N]
        auto [w, h, c, n] = nelements(x);
        tensor t = ggml_reshape_3d(m, x, w * h, c, n);
        t = ggml_mean(m, t);
        pooled = ggml_reshape_4d(m, t, 1, 1, c, n);
    }
    tensor a = conv_2d(m["fc"], pooled);
    a = ggml_hardsigmoid(m, a);
    return ggml_mul(m, x, a);
}

tensor csp_layer(model_ref m, tensor x, int n, bool identity) {
    tensor main = conv_module(m["main_conv"], x);
    tensor shrt = conv_module(m["short_conv"], x);
    for (int i = 0; i < n; ++i) {
        main = cspnext_block(m["blocks"][i], main, identity);
    }
    x = concat(m, {main, shrt}, channel_dim(m));
    x = channel_attention(m["attention"], x);
    return conv_module(m["final_conv"], x);
}

tensor max_pool(model_ref m, tensor x, int k) {
    bool cwhn = is_cwhn(m);
    if (cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    float p = float(k / 2);
    x = ggml_pool_2d(m, x, GGML_OP_POOL_MAX, k, k, 1, 1, p, p);
    if (cwhn) {
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    return x;
}

tensor spp_bottleneck(model_ref m, tensor x) {
    x = conv_module(m["conv1"], x);
    tensor a = max_pool(m, x, 5);
    tensor b = max_pool(m, x, 9);
    tensor c = max_pool(m, x, 13);
    x = concat(m, {x, a, b, c}, channel_dim(m));
    return conv_module(m["conv2"], x);
}

// ScaleNorm: x / rms(x) * g, where rms uses sqrt(dim) scaling -> rms_norm * g.
tensor scale_norm(model_ref m, tensor x) {
    x = ggml_rms_norm(m, x, 1e-5f);
    return ggml_mul(m, x, m.weights("g"));
}

// x: [features, tokens]
tensor gau(model_ref m, tensor x, rtmpose_params const& p) {
    int64_t const e = p.gau_e, s = p.gau_s, t = x->ne[1];
    tensor shortcut = x;
    x = scale_norm(m["ln"], x);
    tensor uv = ggml_silu(m, linear(m["uv"], x)); // [2e + s, t]
    tensor u = ggml_view_2d(m, uv, e, t, uv->nb[1], 0);
    tensor v = ggml_view_2d(m, uv, e, t, uv->nb[1], e * uv->nb[0]);
    tensor base = ggml_view_2d(m, uv, s, t, uv->nb[1], 2 * e * uv->nb[0]);

    tensor gamma = m.weights("gamma"); // [s, 2]
    tensor beta = m.weights("beta");
    auto row = [&](tensor w, int i) { return ggml_view_1d(m, w, s, i * w->nb[1]); };
    tensor q = ggml_add(m, ggml_mul(m, base, row(gamma, 0)), row(beta, 0));
    tensor k = ggml_add(m, ggml_mul(m, base, row(gamma, 1)), row(beta, 1));

    tensor qk = ggml_mul_mat(m, ggml_cont(m, k), ggml_cont(m, q)); // [t_k, t_q]
    qk = ggml_scale(m, qk, 1.f / std::sqrt(float(s)));
    qk = ggml_relu(m, qk);
    qk = ggml_sqr(m, qk);

    tensor vt = ggml_cont(m, ggml_transpose(m, ggml_cont(m, v))); // [t_k, e]
    tensor out = ggml_mul_mat(m, vt, qk);                           // [e, t_q]
    out = ggml_mul(m, ggml_cont(m, u), out);
    out = linear(m["o"], out);
    shortcut = ggml_mul(m, shortcut, m["res_scale"].weights("scale"));
    return ggml_add(m, shortcut, out);
}

tensor rtmcc_head(model_ref m, tensor feat, rtmpose_params const& p) {
    int k = p.final_kernel;
    tensor x = conv_2d(m["final_layer"], feat, 1, k / 2);
    // -> [W*H, K] with token index h*W + w (PyTorch flatten order)
    if (is_cwhn(m)) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    x = ggml_reshape_2d(m, x, x->ne[0] * x->ne[1], x->ne[2]);
    x = scale_norm(m["mlp"][0], x);
    x = linear(m["mlp"][1], x); // [256, K]
    x = gau(m["gau"], x, p);
    tensor sx = linear(m["cls_x"], x); // [W*2, K]
    tensor sy = linear(m["cls_y"], x); // [H*2, K]
    return concat(m, {sx, sy}, 0);    // [W*2 + H*2, K]
}

} // namespace rtmpose

rtmpose_output rtmpose_detect(model_ref m, tensor image, rtmpose_params const& p) {
    using namespace rtmpose;
    model_ref bb = m["backbone"];
    auto const& d = p.stage_depths;

    tensor x = cwhn_to_contiguous_2d(m, image);
    x = conv_module(bb["stem"][0], x, 2);
    x = conv_module(bb["stem"][1], x, 1);
    x = conv_module(bb["stem"][2], x, 1);
    for (int i = 0; i < 3; ++i) {
        model_ref st = bb[format<fixed_string<16>>("stage{}", i + 1).c_str()];
        x = conv_module(st[0], x, 2);
        x = csp_layer(st[1], x, d[i], true);
    }
    model_ref s4 = bb["stage4"];
    x = conv_module(s4[0], x, 2);
    x = spp_bottleneck(s4[1], x);
    x = csp_layer(s4[2], x, d[3], false);

    rtmpose_output r;
    r.simcc = compute_graph_output(m, rtmcc_head(m["head"], x, p), "rtmpose_simcc");
    return r;
}

rtmpose_params rtmpose_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "rtmpose") {
        throw except("Architecture expected to be 'rtmpose', but was '{}' ({})", arch, f.path);
    }
    rtmpose_params p;
    p.input_width = f.get_int("rtmpose.input_width");
    p.input_height = f.get_int("rtmpose.input_height");
    p.num_keypoints = f.get_int("rtmpose.num_keypoints");
    p.final_kernel = f.get_int("rtmpose.final_kernel");
    p.gau_s = f.get_int("rtmpose.gau_s");
    p.gau_e = f.get_int("rtmpose.gau_e");
    f.get_array("rtmpose.stage_depths", span<int>(p.stage_depths.data(), 4));
    if (p.input_width < 32 || p.input_height < 32 || p.input_width % 32 || p.input_height % 32 ||
        p.num_keypoints < 1 || p.num_keypoints > 1024) {
        throw except("RTMPose: invalid input {}x{} or keypoints {}", p.input_width,
                     p.input_height, p.num_keypoints);
    }
    return p;
}

// Affine crop of the (1.25x padded, aspect-fixed) box to the model input, as in mmpose
// TopdownAffine without rotation. Normalised with ImageNet mean/std in 0..255 RGB.
image_data rtmpose_process_input(
    image_view image, std::array<float, 4> box, rtmpose_params const& p, rtmpose_crop* crop) {
    int const W = p.input_width, H = p.input_height;
    float cx = (box[0] + box[2]) * 0.5f, cy = (box[1] + box[3]) * 0.5f;
    float bw = (box[2] - box[0]) * 1.25f, bh = (box[3] - box[1]) * 1.25f;
    float aspect = float(W) / float(H);
    if (bw > bh * aspect) {
        bh = bw / aspect;
    } else {
        bw = bh * aspect;
    }
    if (crop) {
        *crop = {cx, cy, bw, bh};
    }
    image_data rgb = image_u8_to_f32(image, image_format::rgb_f32);
    auto const* src = static_cast<f32x3 const*>(image_view(rgb).data);
    int const iw = image.extent[0], ih = image.extent[1];
    f32x3 const mean = {123.675f, 116.28f, 103.53f};
    f32x3 const stdv = {58.395f, 57.12f, 57.375f};
    image_data out = image_alloc({W, H}, image_format::rgb_f32);
    auto* dst = static_cast<f32x3*>(image_span(out).data);
    auto px = [&](int x, int y) -> f32x3 {
        if (x < 0 || y < 0 || x >= iw || y >= ih) {
            return {0, 0, 0};
        }
        return src[size_t(y) * iw + x];
    };
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            float sx = cx + (float(u) - W * 0.5f) * bw / W;
            float sy = cy + (float(v) - H * 0.5f) * bh / H;
            int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
            float fx = sx - x0, fy = sy - y0;
            f32x3 c = {0, 0, 0};
            for (int j = 0; j < 3; ++j) {
                float a = px(x0, y0)[j] * (1 - fx) + px(x0 + 1, y0)[j] * fx;
                float b = px(x0, y0 + 1)[j] * (1 - fx) + px(x0 + 1, y0 + 1)[j] * fx;
                c[j] = ((a * (1 - fy) + b * fy) * 255.f - mean[j]) / stdv[j];
            }
            dst[size_t(v) * W + u] = c;
        }
    }
    return out;
}

std::vector<rtmpose_keypoint> rtmpose_decode(
    span<float const> simcc, rtmpose_params const& p, rtmpose_crop const& crop) {
    int const nx = p.input_width * 2, ny = p.input_height * 2;
    size_t const row = size_t(nx + ny);
    std::vector<rtmpose_keypoint> kps;
    if (simcc.size() < row * p.num_keypoints) {
        return kps;
    }
    for (int k = 0; k < p.num_keypoints; ++k) {
        float const* v = simcc.data() + k * row;
        auto bx = std::max_element(v, v + nx);
        auto by = std::max_element(v + nx, v + nx + ny);
        float x = float(bx - v) / 2.f, y = float(by - (v + nx)) / 2.f;
        rtmpose_keypoint kp;
        kp.x = crop.cx + (x - p.input_width * 0.5f) * crop.w / p.input_width;
        kp.y = crop.cy + (y - p.input_height * 0.5f) * crop.h / p.input_height;
        kp.score = std::min(*bx, *by);
        kps.push_back(kp);
    }
    return kps;
}

} // namespace visp
