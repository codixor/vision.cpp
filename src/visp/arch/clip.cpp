#include "visp/arch/clip.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/ml.h"
#include "visp/nn.h"

#include <algorithm>
#include <cmath>

namespace visp {
namespace clip {

// Tensor names follow Hugging Face `CLIPVisionModelWithProjection` with the
// `vision_model.` prefix removed (see tools/convert-rs).

tensor embeddings(model_ref m, tensor x, clip_params const& p) {
    // x: [3, w, h, b] (CWHN) -> patches [dim, w/ps, h/ps, b]
    model_ref patch = m["patch_embedding"];
    patch.flags |= model_build_flag::cwhn;
    x = conv_2d(patch, x, p.patch_size);
    x = ggml_reshape_3d(m, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);

    tensor cls = m.weights("class_embedding");
    cls = ggml_reshape_3d(m, cls, cls->ne[0], 1, 1);
    if (x->ne[2] != 1) {
        cls = ggml_repeat_4d(m, cls, cls->ne[0], 1, x->ne[2], 1);
    }
    x = concat(m, {cls, x}, 1);
    x = ggml_add_inplace(m, x, m.weights("position_embedding.weight"));
    return named(m, x);
}

tensor mlp(model_ref m, tensor x, clip_params const& p) {
    x = linear(m["fc1"], x);
    x = p.quick_gelu ? ggml_gelu_quick(m, x) : ggml_gelu_erf(m, x);
    x = linear(m["fc2"], x);
    return x;
}

tensor self_attention(model_ref m, tensor x, clip_params const& p) {
    auto [c, n, b, _] = nelements(x);
    int n_heads = p.n_heads;
    auto project = [&](model_ref m, tensor t) {
        t = linear(m, t);
        return ggml_reshape_4d(m, t, c / n_heads, n_heads, n, b);
    };
    tensor q = project(m["q_proj"], x);
    tensor k = project(m["k_proj"], x);
    tensor v = project(m["v_proj"], x);
    float scale = 1.0f / std::sqrt(float(c) / float(n_heads));
    return attention(m, q, k, v, nullptr, scale, m["out_proj"]);
}

tensor layer(model_ref m, tensor x, clip_params const& p) {
    tensor attn = layer_norm(m["layer_norm1"], x, p.eps);
    attn = self_attention(m["self_attn"], attn, p);
    x = ggml_add(m, x, attn);

    tensor ffn = layer_norm(m["layer_norm2"], x, p.eps);
    ffn = mlp(m["mlp"], ffn, p);
    x = ggml_add(m, x, ffn);
    return named(m, x);
}

clip_output encode(model_ref m, tensor image, clip_params const& p) {
    tensor x = embeddings(m["embeddings"], image, p);
    x = layer_norm(m["pre_layrnorm"], x, p.eps);

    clip_output out{};
    model_ref layers = m["encoder.layers"];
    for (int i = 0; i < p.n_layers; ++i) {
        x = layer(layers[i], x, p);
        if (i == p.n_layers - 2) {
            out.penultimate = ggml_cont(m, x);
            ggml_set_name(out.penultimate, "clip_penultimate");
            ggml_build_forward_expand(m.graph, out.penultimate);
        }
    }
    tensor cls = ggml_cont(m, slice(m, x, {}, {0}, {}, {})); // [dim, 1, b]
    cls = layer_norm(m["post_layernorm"], cls, p.eps);
    tensor embeds = ggml_mul_mat(m, m.weights("visual_projection.weight"), cls);
    out.image_embeds = ggml_reshape_2d(m, embeds, embeds->ne[0], embeds->ne[2]);
    ggml_set_name(out.image_embeds, "clip_image_embeds");
    ggml_build_forward_expand(m.graph, out.image_embeds);
    return out;
}

} // namespace clip

clip_params clip_detect_params(model_file const& file) {
    clip_params p{};
    p.image_size = file.get_int("clip.image_size");
    p.patch_size = file.get_int("clip.patch_size");
    p.embed_dim = file.get_int("clip.embed_dim");
    p.n_heads = file.get_int("clip.n_heads");
    p.n_layers = file.get_int("clip.n_layers");
    p.projection_dim = file.get_int("clip.projection_dim");
    p.quick_gelu = file.get_int("clip.quick_gelu") != 0;
    // Metadata comes from an untrusted file: reject values that would divide by zero,
    // leave the penultimate layer unset or size the graph absurdly.
    if (p.patch_size < 1 || p.image_size < p.patch_size || p.image_size > 4096 ||
        p.image_size % p.patch_size || p.n_heads < 1 || p.embed_dim < p.n_heads ||
        p.embed_dim % p.n_heads || p.n_layers < 2 || p.n_layers > 256 || p.projection_dim < 1) {
        throw except("CLIP: invalid model metadata ({})", file.path);
    }
    return p;
}

image_data clip_process_input(image_view image, clip_params const& p) {
    // Resize so the shorter side equals `image_size`, then center crop (CLIP preprocessing).
    constexpr f32x4 mean = f32x4{0.48145466f, 0.4578275f, 0.40821073f, 0.f};
    constexpr f32x4 std = f32x4{0.26862954f, 0.26130258f, 0.27577711f, 1.f};
    int const s = p.image_size;
    if (image.extent[0] < 1 || image.extent[1] < 1) {
        throw except("CLIP: empty image");
    }
    float const k = float(s) / float(std::min(image.extent[0], image.extent[1]));
    i32x2 scaled = {
        std::max(s, int(std::lround(image.extent[0] * k))),
        std::max(s, int(std::lround(image.extent[1] * k)))};

    image_data resized;
    if (scaled != image.extent) {
        resized = image_scale(image, scaled);
        image = image_view(resized);
    }
    i32x2 offset = {(scaled[0] - s) / 2, (scaled[1] - s) / 2};
    image_data out = image_alloc({s, s}, image_format::rgb_f32);
    image_u8_to_f32(image, image_span(out), -mean, 1.f / std, offset);
    return out;
}

clip_output clip_encode(model_ref m, tensor image, clip_params const& p) {
    return clip::encode(m, image, p);
}

} // namespace visp
