#include "visp/vision.h"
#include "util/math.h"
#include "util/string.h"
#include <cstdlib>

namespace visp {

model_family model_detect_family(model_file const& file) {
    std::string_view arch = file.arch();
    if (arch == "mobile-sam") {
        return model_family::sam;
    } else if (arch == "birefnet") {
        return model_family::birefnet;
    } else if (arch == "depthanything") {
        return model_family::depth_anything;
    } else if (arch == "migan") {
        return model_family::migan;
    } else if (arch == "esrgan") {
        return model_family::esrgan;
    } else if (arch == "clip") {
        return model_family::clip;
    } else if (arch == "yolox") {
        return model_family::yolox;
    } else if (arch == "rtmpose") {
        return model_family::rtmpose;
    } else if (arch == "gfpgan") {
        return model_family::gfpgan;
    } else if (arch == "rife") {
        return model_family::rife;
    } else if (arch == "raft") {
        return model_family::raft;
    }
    return model_family::count;
}

//
// Mobile SAM

sam_model sam_load_model(char const* filepath, backend_device const& dev) {
    sam_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = sam_params{};
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

void sam_encode(sam_model& model, image_view image) {
    if (!model.encoder) {
        model.encoder = compute_graph_init();
        model_ref m = model_ref(model.weights, model.encoder);

        int res = model.params.image_size;
        model.input_image = compute_graph_input(m, GGML_TYPE_F32, {3, res, res, 1});
        tensor embeds = sam_encode_image(m, model.input_image, model.params);
        model.output_embed = compute_graph_output(m, embeds);
        compute_graph_allocate(model.encoder, *model.backend);
    }

    model.image_extent = image.extent;
    image_data img_data = sam_process_input(image, model.params);
    transfer_to_backend(model.input_image, img_data);
    compute(model.encoder, *model.backend);
}

image_data sam_compute_impl(sam_model& model, i32x2 point1, i32x2 point2) {
    ASSERT(model.image_extent[0] > 0, "Missing image embeds, call sam_encode() first");
    bool is_point = point2 == i32x2{-1, -1};

    if (!model.decoder || model.is_point_prompt != is_point) {
        model.is_point_prompt = is_point;

        model.decoder = compute_graph_init();
        model_ref m(model.weights, model.decoder);
        model.input_embed = compute_graph_input(m, GGML_TYPE_F32, {256, 64, 64, 1});
        model.input_prompt = compute_graph_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "input_prompt");
        tensor prompt_embed = is_point ? sam_encode_points(m, model.input_prompt)
                                       : sam_encode_box(m, model.input_prompt);
        model.output = sam_predict_mask(m, model.input_embed, prompt_embed);

        compute_graph_allocate(model.decoder, *model.backend);
    }

    f32x4 prompt_data = is_point
        ? sam_process_point(point1, model.image_extent, model.params)
        : sam_process_box({point1, point2}, model.image_extent, model.params);
    transfer_to_backend(model.input_prompt, span(prompt_data.v, 4));
    ggml_backend_tensor_copy(model.output_embed, model.input_embed);

    compute(model.decoder, *model.backend);

    tensor_data iou_data = transfer_from_backend(model.output.iou);
    tensor_data mask_data = transfer_from_backend(model.output.masks);
    auto iou = iou_data.as_f32().subspan(0, 3);
    int idx = int(std::max_element(iou.begin(), iou.end()) - iou.begin());
    return sam_process_mask(mask_data.as_f32(), idx, model.image_extent, model.params);
}

image_data sam_compute(sam_model& model, i32x2 point) {
    return sam_compute_impl(model, point, i32x2{-1, -1});
}

image_data sam_compute(sam_model& model, box_2d box) {
    return sam_compute_impl(model, box.top_left, box.bottom_right);
}

//
// BiRefNet

birefnet_model birefnet_load_model(char const* filepath, backend_device const& dev) {
    birefnet_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = birefnet_detect_params(file, {1024, 1024});
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data birefnet_compute(birefnet_model& model, image_view image) {
    i32x2 res = birefnet_image_extent(image.extent, model.params, model.backend->max_alloc());
    if (!model.graph || res != model.params.image_extent) {
        model.params.image_extent = res;
        model.graph = compute_graph_init(6 * 1024);

        model_ref m(model.weights, model.graph);
        birefnet_buffers buffers = birefnet_precompute(m, model.params);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, res[0], res[1], 1});
        model.output = birefnet_predict(m, model.input, model.params);

        compute_graph_allocate(model.graph, *model.backend);
        for (tensor_data const& buf : buffers) {
            transfer_to_backend(buf);
        }
    }

    image_data img_data = birefnet_process_input(image, model.params);
    transfer_to_backend(model.input, img_data);

    compute(model.graph, *model.backend);

    tensor_data mask_data = transfer_from_backend(model.output);
    return birefnet_process_output(mask_data.as_f32(), image.extent, model.params);
}

//
// Depth Anything

depthany_model depthany_load_model(char const* filepath, backend_device const& dev) {
    depthany_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = depthany_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data depthany_compute(depthany_model& model, image_view image) {
    i32x2 res = depthany_image_extent(image.extent, model.params);

    if (!model.graph || res != model.params.image_extent) {
        model.params.image_extent = res;
        model.graph = compute_graph_init();

        model_ref m(model.weights, model.graph);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, res[0], res[1], 1});
        model.output = depthany_predict(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }

    image_data img_data = depthany_process_input(image, model.params);
    transfer_to_backend(model.input, img_data);

    compute(model.graph, *model.backend);

    tensor_data output_data = transfer_from_backend(model.output);
    return depthany_process_output(output_data.as_f32(), image.extent, model.params);
}

//
// CLIP vision

clip_model clip_load_model(char const* filepath, backend_device const& dev) {
    clip_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = clip_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

clip_result clip_compute(clip_model& model, image_view image) {
    int const s = model.params.image_size;
    if (!model.graph) {
        model.graph = compute_graph_init();
        model_ref m(model.weights, model.graph);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, s, s, 1});
        model.output = clip_encode(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }
    image_data img_data = clip_process_input(image, model.params);
    transfer_to_backend(model.input, img_data);
    compute(model.graph, *model.backend);

    clip_result r;
    tensor_data embeds = transfer_from_backend(model.output.image_embeds);
    tensor_data hidden = transfer_from_backend(model.output.penultimate);
    r.image_embeds.assign(embeds.as_f32().begin(), embeds.as_f32().end());
    r.penultimate.assign(hidden.as_f32().begin(), hidden.as_f32().end());
    r.n_tokens = int(model.output.penultimate->ne[1]);
    return r;
}

yolox_model yolox_load_model(char const* filepath, backend_device const& dev) {
    yolox_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = yolox_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

std::vector<yolox_box> yolox_compute(
    yolox_model& model, image_view image, float score_thr, float nms_thr) {
    int const s = model.params.input_size;
    if (!model.graph) {
        model.graph = compute_graph_init(8192);
        model_ref m(model.weights, model.graph);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, s, s, 1});
        model.output = yolox_detect(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }
    float ratio = 1;
    image_data img = yolox_process_input(image, model.params, &ratio);
    transfer_to_backend(model.input, img);
    compute(model.graph, *model.backend);
    tensor_data raw = transfer_from_backend(model.output.raw);
    return yolox_decode(raw.as_f32(), model.params, ratio, score_thr, nms_thr);
}

rtmpose_model rtmpose_load_model(char const* filepath, backend_device const& dev) {
    rtmpose_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = rtmpose_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

std::vector<rtmpose_keypoint> rtmpose_compute(
    rtmpose_model& model, image_view image, std::array<float, 4> box) {
    auto const& p = model.params;
    if (!model.graph) {
        model.graph = compute_graph_init(4096);
        model_ref m(model.weights, model.graph);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, p.input_width, p.input_height, 1});
        model.output = rtmpose_detect(m, model.input, p);
        compute_graph_allocate(model.graph, *model.backend);
    }
    rtmpose_crop crop;
    image_data img = rtmpose_process_input(image, box, p, &crop);
    transfer_to_backend(model.input, img);
    compute(model.graph, *model.backend);
    tensor_data simcc = transfer_from_backend(model.output.simcc);
    return rtmpose_decode(simcc.as_f32(), p, crop);
}

gfpgan_model gfpgan_load_model(char const* filepath, backend_device const& dev) {
    gfpgan_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = gfpgan_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), tensor_data_layout::whcn);
    return model;
}

image_data gfpgan_compute(gfpgan_model& model, image_view face) {
    int const s = model.params.image_size;
    if (!model.graph) {
        model.graph = compute_graph_init(4096);
        model_ref m(model.weights, model.graph);
        model.input = compute_graph_input(m, GGML_TYPE_F32, {3, s, s, 1});
        model.output = gfpgan_restore(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }
    image_data resized;
    if (face.extent != i32x2{s, s}) {
        resized = image_scale(face, {s, s});
        face = image_view(resized);
    }
    image_data input = image_u8_to_f32(face, image_format::rgb_f32);
    transfer_to_backend(model.input, input);
    compute(model.graph, *model.backend);
    image_data output = image_alloc({s, s}, image_format::rgb_f32);
    transfer_from_backend(model.output, output);
    return image_f32_to_u8(output, image_format::rgba_u8);
}

//
// RIFE

rife_model rife_load_model(char const* filepath, backend_device const& dev) {
    rife_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = rife_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), tensor_data_layout::whcn);
    return model;
}

image_data rife_process_input(image_view frame) {
    int const a = rife_params::alignment;
    i32x2 padded{(frame.extent[0] + a - 1) / a * a, (frame.extent[1] + a - 1) / a * a};
    image_data f = image_u8_to_f32(frame, image_format::rgb_f32);
    if (padded == frame.extent) {
        return f;
    }
    image_data out = image_alloc(padded, image_format::rgb_f32);
    image_clear(out);
    float const* src = image_view(f).as_floats().data();
    float* dst = image_span(out).as_floats().data();
    for (int y = 0; y < frame.extent[1]; ++y) {
        std::copy_n(src + size_t(y) * frame.extent[0] * 3, size_t(frame.extent[0]) * 3,
                    dst + size_t(y) * padded[0] * 3);
    }
    return out;
}

image_data rife_compute(rife_model& model, image_view frame0, image_view frame1, float t) {
    if (frame0.extent != frame1.extent) {
        throw except("RIFE: frames must have the same size");
    }
    image_data in0 = rife_process_input(frame0);
    image_data in1 = rife_process_input(frame1);
    if (!model.graph || model.extent != in0.extent || model.t != t) {
        model.graph = compute_graph_init(8192);
        model_ref m(model.weights, model.graph);
        model.input0 = compute_graph_input(m, GGML_TYPE_F32, {3, in0.extent[0], in0.extent[1], 1});
        model.input1 = compute_graph_input(m, GGML_TYPE_F32, {3, in0.extent[0], in0.extent[1], 1});
        model.output = rife_interpolate(m, model.input0, model.input1, t, model.params);
        compute_graph_allocate(model.graph, *model.backend);
        model.extent = in0.extent;
        model.t = t;
    }
    transfer_to_backend(model.input0, in0);
    transfer_to_backend(model.input1, in1);
    compute(model.graph, *model.backend);
    image_data out = image_alloc(in0.extent, image_format::rgb_f32);
    transfer_from_backend(model.output, out);
    image_data u8 = image_f32_to_u8(out, image_format::rgba_u8);
    if (u8.extent == frame0.extent) {
        return u8;
    }
    image_data crop = image_alloc(frame0.extent, image_format::rgba_u8);
    for (int y = 0; y < frame0.extent[1]; ++y) {
        std::copy_n(u8.data.get() + size_t(y) * u8.extent[0] * 4, size_t(frame0.extent[0]) * 4,
                    crop.data.get() + size_t(y) * frame0.extent[0] * 4);
    }
    return crop;
}

//
// RAFT

raft_model raft_load_model(char const* filepath, backend_device const& dev) {
    raft_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = raft_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), tensor_data_layout::whcn);
    return model;
}

raft_flow_field raft_compute(raft_model& model, image_view frame0, image_view frame1) {
    if (frame0.extent != frame1.extent) {
        throw except("RAFT: frames must have the same size");
    }
    int const a = raft_params::alignment;
    i32x2 const size{std::max(128, (frame0.extent[0] + a - 1) / a * a),
                     std::max(128, (frame0.extent[1] + a - 1) / a * a)};
    auto prepare = [&](image_view frame) {
        image_data f = image_u8_to_f32(frame, image_format::rgb_f32);
        if (size == frame.extent) {
            return f;
        }
        image_data out = image_alloc(size, image_format::rgb_f32);
        image_clear(out);
        float const* src = image_view(f).as_floats().data();
        float* dst = image_span(out).as_floats().data();
        for (int y = 0; y < frame.extent[1]; ++y) {
            std::copy_n(src + size_t(y) * frame.extent[0] * 3, size_t(frame.extent[0]) * 3,
                        dst + size_t(y) * size[0] * 3);
        }
        return out;
    };
    image_data in0 = prepare(frame0);
    image_data in1 = prepare(frame1);
    if (!model.graph || model.extent != size) {
        model.graph = compute_graph_init(16384);
        model_ref m(model.weights, model.graph);
        model.input0 = compute_graph_input(m, GGML_TYPE_F32, {3, size[0], size[1], 1});
        model.input1 = compute_graph_input(m, GGML_TYPE_F32, {3, size[0], size[1], 1});
        model.output = raft_flow(m, model.input0, model.input1, model.params);
        compute_graph_allocate(model.graph, *model.backend);
        model.extent = size;
    }
    transfer_to_backend(model.input0, in0);
    transfer_to_backend(model.input1, in1);
    compute(model.graph, *model.backend);
    std::vector<float> planes(size_t(size[0]) * size[1] * 2);
    transfer_from_backend(model.output, span<float>(planes.data(), planes.size()));

    raft_flow_field field;
    field.extent = frame0.extent;
    size_t const n = size_t(frame0.extent[0]) * frame0.extent[1];
    field.u.resize(n);
    field.v.resize(n);
    size_t const plane = size_t(size[0]) * size[1];
    for (int y = 0; y < frame0.extent[1]; ++y) {
        for (int x = 0; x < frame0.extent[0]; ++x) {
            size_t i = size_t(y) * size[0] + x, o = size_t(y) * frame0.extent[0] + x;
            field.u[o] = planes[i];
            field.v[o] = planes[plane + i];
        }
    }
    return field;
}

image_data raft_flow_to_image(raft_flow_field const& f, float max_flow) {
    // Baker et al. color wheel (RY 15, YG 6, GC 4, CB 11, BM 13, MR 6)
    std::vector<std::array<float, 3>> wheel;
    auto ramp = [&](int n, std::array<float, 3> a, std::array<float, 3> b) {
        for (int i = 0; i < n; ++i) {
            float t = float(i) / float(n);
            wheel.push_back({a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t});
        }
    };
    ramp(15, {255, 0, 0}, {255, 255, 0});
    ramp(6, {255, 255, 0}, {0, 255, 0});
    ramp(4, {0, 255, 0}, {0, 255, 255});
    ramp(11, {0, 255, 255}, {0, 0, 255});
    ramp(13, {0, 0, 255}, {255, 0, 255});
    ramp(6, {255, 0, 255}, {255, 0, 0});
    int const ncols = int(wheel.size());

    size_t const n = f.u.size();
    if (max_flow <= 0.f) {
        for (size_t i = 0; i < n; ++i) {
            max_flow = std::max(max_flow, std::sqrt(f.u[i] * f.u[i] + f.v[i] * f.v[i]));
        }
        max_flow = std::max(max_flow, 1e-5f);
    }
    image_data out = image_alloc(f.extent, image_format::rgba_u8);
    uint8_t* px = out.data.get();
    for (size_t i = 0; i < n; ++i) {
        float u = f.u[i] / max_flow, v = f.v[i] / max_flow;
        float rad = std::min(1.f, std::sqrt(u * u + v * v));
        float a = std::atan2(-v, -u) / 3.14159265f;
        float fk = (a + 1.f) / 2.f * float(ncols - 1);
        int k0 = int(std::floor(fk)), k1 = (k0 + 1) % ncols;
        float t = fk - float(k0);
        for (int c = 0; c < 3; ++c) {
            float col = ((1.f - t) * wheel[k0][c] + t * wheel[k1][c]) / 255.f;
            col = 1.f - rad * (1.f - col);
            px[i * 4 + c] = uint8_t(std::lround(255.f * col));
        }
        px[i * 4 + 3] = 255;
    }
    return out;
}

//
// MI-GAN

migan_model migan_load_model(char const* filepath, backend_device const& dev) {
    migan_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = migan_detect_params(file);
    model.params.invert_mask = true; // inpaint opaque areas
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data migan_compute(migan_model& model, image_view image, image_view mask) {
    if (!model.graph) {
        model.graph = compute_graph_init();
        model_ref m(model.weights, model.graph);

        int res = model.params.resolution;
        model.input = compute_graph_input(m, GGML_TYPE_F32, {4, res, res, 1});
        model.output = migan_generate(m, model.input, model.params);
        compute_graph_allocate(model.graph, *model.backend);
    }

    image_data input_data = migan_process_input(image, mask, model.params);
    transfer_to_backend(model.input, input_data);

    compute(model.graph, *model.backend);

    tensor_data output_data = transfer_from_backend(model.output);
    image_data output = migan_process_output(output_data.as_f32(), image.extent, model.params);
    image_set_alpha(output, mask);
    return output;
}

//
// ESRGAN

constexpr int esrgan_default_tile_size = 224;

esrgan_model esrgan_load_model(char const* filepath, backend_device const& dev) {
    esrgan_model model;
    model.backend = &dev;
    model_file file = model_load(filepath);
    model.params = esrgan_detect_params(file);
    model.weights = model_init(file.n_tensors());
    model_transfer(file, model.weights, dev, dev.preferred_float_type(), dev.preferred_layout());
    return model;
}

image_data esrgan_compute(esrgan_model& model, image_view image) {
    tile_layout tiles(image.extent, esrgan_default_tile_size, 16);
    if (!model.graph || model.tile_size != tiles.tile_size) {
        model.tile_size = tiles.tile_size;
        model.graph = compute_graph_init(esrgan_estimate_graph_size(model.params));

        model_ref m(model.weights, model.graph);
        i64x4 input_shape = {3, tiles.tile_size[0], tiles.tile_size[1], 1};
        model.input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
        model.output = esrgan_generate(m, model.input, model.params);

        compute_graph_allocate(model.graph, *model.backend);
    }

    tile_layout tiles_out = tile_scale(tiles, model.params.scale);
    image_data input_tile = image_alloc(tiles.tile_size, image_format::rgb_f32);
    image_data output_tile = image_alloc(tiles_out.tile_size, image_format::rgb_f32);
    image_data output_image = image_alloc(image.extent * model.params.scale, image_format::rgb_f32);
    image_clear(output_image);

    for (int t = 0; t < tiles.total(); ++t) {
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);

        image_u8_to_f32(image, input_tile, f32x4{0, 0, 0, 0}, f32x4{1, 1, 1, 1}, tile_offset);
        transfer_to_backend(model.input, input_tile);

        compute(model.graph, *model.backend);

        transfer_from_backend(model.output, output_tile);
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }
    return image_f32_to_u8(output_image, image_format::rgba_u8);
}

} // namespace visp