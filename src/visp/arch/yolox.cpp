// YOLOX object detector (Megvii, Apache-2.0). Standard (non-depthwise) variants S/M/L/X.
// Weights: official .pth -> pickle2safetensors --key model -> visp-convert yolox.
// Batch norm is fused into the convolutions by the converter; the Focus stem is rewritten
// as an equivalent 6x6 stride-2 convolution (padding 2).

#include "visp/arch/yolox.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace visp {
namespace yolox {

int channel_dim(model_ref const& m) {
    return (m.flags & model_build_flag::cwhn) ? 0 : 2;
}

// Conv + (fused BN) + SiLU, "same" padding.
tensor base_conv(model_ref m, tensor x, int stride) {
    tensor w = m["conv"].weights("weight");
    // CWHN weights are [cin, kw, kh, cout]; WHCN weights are [kw, kh, cin, cout].
    int k = (m.flags & model_build_flag::cwhn) ? int(w->ne[1]) : int(w->ne[0]);
    x = conv_2d(m["conv"], x, stride, (k - 1) / 2);
    return ggml_silu_inplace(m, x);
}

tensor bottleneck(model_ref m, tensor x, bool shortcut) {
    tensor y = base_conv(m["conv1"], x, 1);
    y = base_conv(m["conv2"], y, 1);
    return shortcut ? ggml_add(m, y, x) : y;
}

tensor csp_layer(model_ref m, tensor x, int n, bool shortcut) {
    tensor x1 = base_conv(m["conv1"], x, 1);
    tensor x2 = base_conv(m["conv2"], x, 1);
    for (int i = 0; i < n; ++i) {
        x1 = bottleneck(m["m"][i], x1, shortcut);
    }
    x = concat(m, {x1, x2}, channel_dim(m));
    return base_conv(m["conv3"], x, 1);
}

tensor max_pool(model_ref m, tensor x, int k) {
    bool cwhn = bool(m.flags & model_build_flag::cwhn);
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

tensor spp(model_ref m, tensor x) {
    x = base_conv(m["conv1"], x, 1);
    tensor a = max_pool(m, x, 5);
    tensor b = max_pool(m, x, 9);
    tensor c = max_pool(m, x, 13);
    x = concat(m, {x, a, b, c}, channel_dim(m));
    return base_conv(m["conv2"], x, 1);
}

tensor upsample2(model_ref m, tensor x) {
    auto [w, h, c, n] = nelements_whcn(m, x);
    return interpolate(m, x, {w * 2, h * 2}, GGML_SCALE_MODE_NEAREST);
}

// One scale of the decoupled head -> [5 + classes, w*h] raw values per grid cell
// (box offsets, objectness logit, class logits), cells ordered row-major (y, then x).
tensor head_level(model_ref m, int i, tensor x) {
    model_ref h = m["head"];
    x = base_conv(h["stems"][i], x, 1);
    tensor c = base_conv(h["cls_convs"][i][1], base_conv(h["cls_convs"][i][0], x, 1), 1);
    tensor r = base_conv(h["reg_convs"][i][1], base_conv(h["reg_convs"][i][0], x, 1), 1);
    tensor reg = conv_2d(h["reg_preds"][i], r);
    tensor obj = conv_2d(h["obj_preds"][i], r);
    tensor cls = conv_2d(h["cls_preds"][i], c);
    tensor out = concat(m, {reg, obj, cls}, channel_dim(m));
    if (!(m.flags & model_build_flag::cwhn)) { // [W, H, C, N] -> [C, W, H, N]
        out = ggml_cont(m, permute_whcn_to_cwhn(m, out));
    }
    return ggml_reshape_2d(m, out, out->ne[0], out->ne[1] * out->ne[2]);
}

} // namespace yolox

yolox_output yolox_detect(model_ref m, tensor image, yolox_params const& p) {
    using namespace yolox;
    int const d = p.depth;
    model_ref pf = m["backbone"];
    model_ref bb = pf["backbone"];

    tensor x = cwhn_to_contiguous_2d(m, image);
    x = base_conv(bb["stem"]["conv"], x, 2);
    x = base_conv(bb["dark2"][0], x, 2);
    x = csp_layer(bb["dark2"][1], x, d, true);
    x = base_conv(bb["dark3"][0], x, 2);
    tensor c3 = csp_layer(bb["dark3"][1], x, 3 * d, true);
    x = base_conv(bb["dark4"][0], c3, 2);
    tensor c4 = csp_layer(bb["dark4"][1], x, 3 * d, true);
    x = base_conv(bb["dark5"][0], c4, 2);
    x = spp(bb["dark5"][1], x);
    tensor c5 = csp_layer(bb["dark5"][2], x, d, false);

    int cd = channel_dim(m);
    tensor fpn0 = base_conv(pf["lateral_conv0"], c5, 1);
    x = concat(m, {upsample2(m, fpn0), c4}, cd);
    x = csp_layer(pf["C3_p4"], x, d, false);
    tensor fpn1 = base_conv(pf["reduce_conv1"], x, 1);
    x = concat(m, {upsample2(m, fpn1), c3}, cd);
    tensor p3 = csp_layer(pf["C3_p3"], x, d, false);
    x = base_conv(pf["bu_conv2"], p3, 2);
    x = concat(m, {x, fpn1}, cd);
    tensor p4 = csp_layer(pf["C3_n3"], x, d, false);
    x = base_conv(pf["bu_conv1"], p4, 2);
    x = concat(m, {x, fpn0}, cd);
    tensor p5 = csp_layer(pf["C3_n4"], x, d, false);

    tensor out = concat(m, {head_level(m, 0, p3), head_level(m, 1, p4), head_level(m, 2, p5)}, 1);
    yolox_output r;
    r.raw = compute_graph_output(m, out, "yolox_raw");
    return r;
}

yolox_params yolox_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "yolox") {
        throw except("Architecture expected to be 'yolox', but was '{}' ({})", arch, f.path);
    }
    yolox_params p;
    p.depth = f.get_int("yolox.depth");
    p.num_classes = f.get_int("yolox.num_classes");
    if (p.depth < 1 || p.depth > 8 || p.num_classes < 1 || p.num_classes > 4096) {
        throw except("YOLOX: invalid depth {} or classes {}", p.depth, p.num_classes);
    }
    return p;
}

image_data yolox_process_input(image_view image, yolox_params const& p, float* ratio) {
    // Official YOLOX preprocessing: letterbox to input_size (pad bottom/right with 114),
    // BGR channel order, raw 0..255 values, no normalisation.
    int const s = p.input_size;
    if (s < 32 || s > 4096 || s % 32) {
        throw except("YOLOX: input size must be a multiple of 32 in 32..4096, got {}", s);
    }
    if (image.extent[0] < 1 || image.extent[1] < 1) {
        throw except("YOLOX: empty image");
    }
    float r = std::min(float(s) / float(image.extent[0]), float(s) / float(image.extent[1]));
    i32x2 scaled = {std::max(1, int(image.extent[0] * r)), std::max(1, int(image.extent[1] * r))};
    image_data resized;
    if (scaled != image.extent) {
        resized = image_scale(image, scaled);
        image = image_view(resized);
    }
    image_data rgb = image_alloc(scaled, image_format::rgb_f32);
    image_u8_to_f32(image, image_span(rgb), f32x4{0, 0, 0, 0}, f32x4{1, 1, 1, 1}, i32x2{0, 0});
    image_data out = image_alloc({s, s}, image_format::rgb_f32);
    auto const* src = static_cast<f32x3 const*>(image_view(rgb).data);
    auto* dst = static_cast<f32x3*>(image_span(out).data);
    for (int y = 0; y < s; ++y) {
        for (int x = 0; x < s; ++x) {
            f32x3 v = {114.f, 114.f, 114.f};
            if (x < scaled[0] && y < scaled[1]) {
                f32x3 c = src[size_t(y) * scaled[0] + x];
                v = {c[2] * 255.f, c[1] * 255.f, c[0] * 255.f};
            }
            dst[size_t(y) * s + x] = v;
        }
    }
    if (ratio) {
        *ratio = r;
    }
    return out;
}

std::vector<yolox_box> yolox_decode(
    span<float const> raw, yolox_params const& p, float ratio, float score_thr, float nms_thr) {
    int const s = p.input_size;
    size_t const n_ch = size_t(5 + p.num_classes);
    std::vector<yolox_box> boxes;
    size_t row = 0;
    for (int stride : {8, 16, 32}) {
        int g = s / stride;
        for (int gy = 0; gy < g; ++gy) {
            for (int gx = 0; gx < g; ++gx, ++row) {
                if ((row + 1) * n_ch > raw.size()) {
                    return {};
                }
                float const* v = raw.data() + row * n_ch;
                int best = 0;
                for (int c = 1; c < p.num_classes; ++c) {
                    if (v[5 + c] > v[5 + best]) {
                        best = c;
                    }
                }
                float obj = 1.f / (1.f + std::exp(-v[4]));
                float score = obj / (1.f + std::exp(-v[5 + best]));
                if (!(score >= score_thr)) { // also drops NaN, which would break std::sort
                    continue;
                }
                float cx = (v[0] + float(gx)) * float(stride);
                float cy = (v[1] + float(gy)) * float(stride);
                float w = std::exp(v[2]) * float(stride);
                float h = std::exp(v[3]) * float(stride);
                yolox_box b;
                b.box = {(cx - w / 2) / ratio, (cy - h / 2) / ratio, (cx + w / 2) / ratio,
                         (cy + h / 2) / ratio};
                b.score = score;
                b.label = best;
                boxes.push_back(b);
            }
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](auto const& a, auto const& b) {
        return a.score > b.score;
    });
    auto iou = [](yolox_box const& a, yolox_box const& b) {
        float ix = std::max(0.f, std::min(a.box[2], b.box[2]) - std::max(a.box[0], b.box[0]));
        float iy = std::max(0.f, std::min(a.box[3], b.box[3]) - std::max(a.box[1], b.box[1]));
        float in = ix * iy;
        float un = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) +
                   (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - in;
        return un > 0 ? in / un : 0.f;
    };
    std::vector<yolox_box> keep;
    for (auto const& b : boxes) {
        bool ok = std::none_of(keep.begin(), keep.end(), [&](yolox_box const& k) {
            return k.label == b.label && iou(k, b) > nms_thr;
        });
        if (ok) {
            keep.push_back(b);
        }
    }
    return keep;
}

} // namespace visp
