// GFPGAN v1.3/v1.4 "clean" face restoration (Tencent ARC, Apache-2.0). U-Net encoder whose
// features modulate a StyleGAN2-style decoder via SFT. The "clean" variant uses plain PyTorch ops
// (no NVIDIA StyleGAN2 CUDA kernels). Runs in WHCN layout (bilinear resampling).
// Weights: official .pth -> pickle2safetensors --key params_ema -> visp-convert gfpgan.
// Input: an aligned 512x512 face crop.

#include "visp/arch/gfpgan.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <cmath>
#include <string_view>

namespace visp {
namespace gfpgan {

tensor lrelu(model_ref m, tensor x) {
    return ggml_leaky_relu(m, x, 0.2f, true);
}

tensor conv(model_ref m, tensor x) {
    int k = int(m.weights("weight")->ne[0]);
    return conv_2d(m, x, 1, k / 2);
}

tensor resize(model_ref m, tensor x, float f) {
    auto w = int64_t(float(x->ne[0]) * f), h = int64_t(float(x->ne[1]) * f);
    return ggml_interpolate(m, x, w, h, x->ne[2], x->ne[3], GGML_SCALE_MODE_BILINEAR);
}

tensor res_block(model_ref m, tensor x, float f) {
    tensor y = lrelu(m, conv(m["conv1"], x));
    y = resize(m, y, f);
    y = lrelu(m, conv(m["conv2"], y));
    tensor s = conv_2d(m["skip"], resize(m, x, f));
    return ggml_add(m, y, s);
}

tensor bias_4d(model_ref m, tensor b) {
    return ggml_reshape_4d(m, b, 1, 1, b->ne[0], 1);
}

// StyleGAN2 modulated conv for batch 1: scaling the input channels by the style and the output
// channels by the demodulation factor is equivalent to modulating the kernel.
tensor modulated_conv(model_ref m, tensor x, tensor latent, bool demodulate) {
    tensor s = linear(m["modulation"], latent); // [cin]
    int k = int(m.weights("weight")->ne[0]);
    x = ggml_mul(m, x, bias_4d(m, s));
    x = conv_2d(m, x, 1, k / 2);
    if (demodulate) {
        // d = 1 / sqrt(sum_{cin,k} (w * s)^2 + eps), with wsq = sum_k w^2 precomputed [cin, cout]
        tensor d = ggml_mul_mat(m, m.weights("wsq"), ggml_sqr(m, s)); // [cout]
        d = ggml_sqrt(m, ggml_scale_bias(m, d, 1.f, 1e-8f));
        tensor one = ggml_scale_bias(m, d, 0.f, 1.f);
        x = ggml_mul(m, x, bias_4d(m, ggml_div(m, one, d)));
    }
    return x;
}

tensor style_conv(model_ref m, tensor x, tensor latent, bool upsample) {
    if (upsample) {
        x = resize(m, x, 2.f);
    }
    x = modulated_conv(m["mod"], x, latent, true);
    x = ggml_scale(m, x, std::sqrt(2.f));
    tensor noise = m.weights("noise");
    x = ggml_add(m, x, ggml_reshape_4d(m, noise, noise->ne[0], noise->ne[1], 1, 1));
    x = ggml_add(m, x, bias_4d(m, m.weights("bias")));
    return lrelu(m, x);
}

tensor to_rgb(model_ref m, tensor x, tensor latent, tensor skip) {
    x = modulated_conv(m["mod"], x, latent, false);
    x = ggml_add(m, x, bias_4d(m, m.weights("bias")));
    if (skip) {
        x = ggml_add(m, x, resize(m, skip, 2.f));
    }
    return x;
}

// SFT on the second half of the channels: [same, sft * scale + shift]
tensor sft_half(model_ref m, tensor x, tensor scale, tensor shift) {
    int64_t half = x->ne[2] / 2;
    tensor a = ggml_view_4d(m, x, x->ne[0], x->ne[1], half, x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
    tensor b = ggml_view_4d(
        m, x, x->ne[0], x->ne[1], half, x->ne[3], x->nb[1], x->nb[2], x->nb[3], half * x->nb[2]);
    b = ggml_add(m, ggml_mul(m, b, scale), shift);
    return ggml_concat(m, a, b, 2);
}

} // namespace gfpgan

tensor gfpgan_restore(model_ref m, tensor image, gfpgan_params const& p) {
    using namespace gfpgan;
    int const levels = p.levels();

    tensor x = cwhn_to_contiguous_2d(m, image);
    x = ggml_scale_bias(m, x, 2.f, -1.f); // [0, 1] -> [-1, 1]
    tensor feat = lrelu(m, conv_2d(m["conv_body_first"], x));
    std::vector<tensor> skips;
    for (int i = 0; i < levels; ++i) {
        feat = res_block(m["conv_body_down"][i], feat, 0.5f);
        skips.insert(skips.begin(), feat);
    }
    feat = lrelu(m, conv(m["final_conv"], feat));

    // PyTorch flattens [C, H, W]; WHCN memory order is the same (w fastest, then h, then c).
    tensor flat = ggml_reshape_1d(m, ggml_cont(m, feat), ggml_nelements(feat));
    tensor style = linear(m["final_linear"], flat); // [style_dim * num_latent]
    auto latent = [&](int i) {
        return ggml_view_1d(m, style, p.style_dim, size_t(i) * p.style_dim * ggml_element_size(style));
    };

    std::vector<tensor> conditions;
    for (int i = 0; i < levels; ++i) {
        feat = ggml_add(m, feat, skips[i]);
        feat = res_block(m["conv_body_up"][i], feat, 2.f);
        model_ref cs = m["condition_scale"][i];
        model_ref ch = m["condition_shift"][i];
        conditions.push_back(conv(cs[2], lrelu(m, conv(cs[0], feat))));
        conditions.push_back(conv(ch[2], lrelu(m, conv(ch[0], feat))));
    }

    model_ref dec = m["dec"];
    tensor c = dec["constant_input"].weights("weight");
    tensor out = ggml_reshape_4d(m, c, c->ne[0], c->ne[1], c->ne[2], 1);
    if (out->type != GGML_TYPE_F32) { // weights may be stored as F16 on GPU backends
        out = ggml_cast(m, out, GGML_TYPE_F32);
    }
    out = style_conv(dec["style_conv1"], out, latent(0), false);
    tensor skip = to_rgb(dec["to_rgb1"], out, latent(1), nullptr);
    for (int j = 0, i = 1; j < levels; ++j, i += 2) {
        out = style_conv(dec["style_convs"][2 * j], out, latent(i), true);
        out = sft_half(m, out, conditions[2 * j], conditions[2 * j + 1]);
        out = style_conv(dec["style_convs"][2 * j + 1], out, latent(i + 1), false);
        skip = to_rgb(dec["to_rgbs"][j], out, latent(i + 2), skip);
    }
    x = ggml_clamp(m, ggml_scale_bias(m, skip, 0.5f, 0.5f), 0.f, 1.f); // [-1, 1] -> [0, 1]
    x = contiguous_2d_to_cwhn(m, x);
    return compute_graph_output(m, x, "gfpgan_result");
}

gfpgan_params gfpgan_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "gfpgan") {
        throw except("Architecture expected to be 'gfpgan', but was '{}' ({})", arch, f.path);
    }
    gfpgan_params p;
    p.image_size = f.get_int("gfpgan.image_size");
    p.style_dim = f.get_int("gfpgan.style_dim");
    p.num_latent = f.get_int("gfpgan.num_latent");
    if (p.image_size < 64 || p.image_size > 1024 || (p.image_size & (p.image_size - 1)) ||
        p.num_latent != p.levels() * 2 + 2) {
        throw except("GFPGAN: unsupported size {} / latents {}", p.image_size, p.num_latent);
    }
    return p;
}

} // namespace visp
