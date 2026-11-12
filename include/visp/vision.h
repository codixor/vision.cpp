//
// Vision.cpp
//
// Library for inference of computer vision neural networks.
//
// Overview
// --------
//
// Vision.cpp comes in 3 main headers:
//
// visp/image.h
//
//   Defines structures to store and reference pixel data. Supports loading, saving and
//   common processing of images. Most tasks take an `image_view` as input, which
//   is a non-owning reference to external pixel data. Output is returned as
//   `image_data` (allocated by the library) or written to an `image_span`.
//
// visp/ml.h
//
//   Contains ML infrastructure shared between all models: loading weights,
//   transferring data between backend devices (eg. GPU), and executing
//   compute graphs. Most of these are thin convenience wrappers around GGML.
//   Alternatively you can use GGML directly for greater flexibility.
//
// visp/vision.h (this file)
//
//   Provides a high-level API to run inference on various vision models for
//   common tasks. These operations are built for simplicity and don't provide
//   a lot of options. If you need more control, you will find each operation
//   split into several steps below, which can be combined in a modular fashion.
//
// Basic Use
// ---------
//
// Evaluating a model on an image with the high-level API usually looks like this:
//
// ```
// backend_device backend = backend_init();
// image_view input  = image_view({width, height}, image_format::rgb_u8, pixel_data_ptr);
// ARCH_model model  = ARCH_load_model("path/to/model.gguf", backend);
// image_data output = ARCH_compute(model, input);
// ```
//
// ARCH referes to the model architecture, such as `sam_model`, `birefnet_model`, etc.
// The `compute` function can be called repeatedly with different inputs.
//
// Advanced Use
// ------------
//
// Internally running the model is split into several steps:
// 1. Load the model weights from a GGUF file.
// 2. Detect model hyperparameters and precompute required buffers.
// 3. Allocate storage on the backend device and transfer the weights.
// 4. Build a compute graph for the model architecture.
// 5. Allocate storage for input, output and intermediate tensors on the backend device.
// 6. Pre-process the input and transfer it to the backend device.
// 7. Run the compute graph.
// 8. Transfer the output to the host and post-process it.
//
// Custom pipelines can be created simply by writing a function that calls the
// individual steps. As a starting point, check out or copy the implementation
// of the high-level API functions. Then adapt them as needed.
// This allows to:
// * load model weights from a different source
// * control exactly when allocation happens
// * offload weights to host memory when not needed
// * implement pre-processing and post-processing for custom image formats
// * integrate model compute graphs into your own GGML graphs
// * ... etc.
//

#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"

#include <array>
#include <span>
#include <vector>

namespace visp {

// Supported model architectures

enum class model_family {
    sam = 0,
    birefnet,
    depth_anything,
    migan,
    esrgan,
    clip,
    yolox,
    rtmpose,
    gfpgan,
    rife,

    count
};

VISP_API model_family model_detect_family(model_file const&);    

//
// SWIN v1 - vision transformer for feature extraction

constexpr int swin_n_layers = 4;

struct swin_layer_t {
    int depth;
    int n_heads;
    int n_features;
};

struct swin_params {
    int embed_dim;
    int window_size;
    std::array<swin_layer_t, swin_n_layers> layers;
};

using swin_buffers = std::array<tensor_data, swin_n_layers + 2>;
using swin_result = std::array<tensor, swin_n_layers>;

VISP_API swin_params swin_detect_params(model_file const&);
VISP_API swin_buffers swin_precompute(model_ref, i32x2 image_extent, swin_params const&);
VISP_API swin_result swin_encode(model_ref, tensor image, swin_params const&);

//
// DINO v2 - vision transformer for feature extraction

struct dino_params {
    int patch_size = 16;
    int embed_dim = 768;
    int n_layers = 12;
    int n_heads = 12;
};

VISP_API dino_params dino_detect_params(model_file const&);
VISP_API std::vector<tensor> dino_get_intermediate_layers(
    model_ref, tensor image, span<int const> layers_ids, dino_params const&);

//
// Mobile SAM - image segmentation with prompt (point or box)

struct sam_model;

struct box_2d {
    i32x2 top_left;
    i32x2 bottom_right;
};

// Loads a SAM model from GGUF file onto the backend device.
// * only supports MobileSAM (TinyViT) for now
VISP_API sam_model sam_load_model(char const* filepath, backend_device const&);

// Creates an image embedding from RGB input, required for subsequent `sam_compute` calls.
VISP_API void sam_encode(sam_model&, image_view image);

// Computes a segmentation mask (alpha image) for an object in the image.
// * takes either a point, ie. a pixel location with origin (0, 0) in the top left
// * or a bounding box which contains the object
VISP_API image_data sam_compute(sam_model&, i32x2 point);
VISP_API image_data sam_compute(sam_model&, box_2d box);

// --- SAM pipeline

struct sam_params {
    int image_size = 1024;
    int mask_size = 256;
};

struct sam_prediction {
    tensor masks;
    tensor iou;
};

VISP_API image_data sam_process_input(image_view image, sam_params const&);
VISP_API f32x4 sam_process_point(i32x2 point, i32x2 image_extent, sam_params const&);
VISP_API f32x4 sam_process_box(box_2d box, i32x2 image_extent, sam_params const&);

VISP_API tensor sam_encode_image(model_ref, tensor image, sam_params const&);
VISP_API tensor sam_encode_points(model_ref, tensor coords);
VISP_API tensor sam_encode_box(model_ref, tensor coords);

VISP_API sam_prediction sam_predict_mask(model_ref m, tensor image_embed, tensor prompt_embed);

VISP_API image_data sam_process_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const&);

//
// BiRefNet - dichotomous image segmentation (background removal)

struct birefnet_model;

// Loads a BiRefNet model from GGUF file onto the backend device.
// * supports BiRefNet, BiRefNet-lite, BiRefNet-Matting variants at 1024px resolution
// * supports BiRefNet-HR variant at 2048px resolution
// * supports BiRefNet-dynamic variant at arbitrary resolution
VISP_API birefnet_model birefnet_load_model(char const* filepath, backend_device const&);

// Takes RGB input and computes an alpha mask with foreground as 1.0 and background as 0.0.
VISP_API image_data birefnet_compute(birefnet_model&, image_view image);

// --- BiRefNet pipeline

struct birefnet_params {
    int image_size = 1024; // can be -1 for dynamic size
    int image_multiple = 32;
    i32x2 image_extent = {1024, 1024}; // required if image_size is -1
    swin_params encoder;
};

using birefnet_buffers = swin_buffers;

VISP_API birefnet_params birefnet_detect_params(
    model_file const&, i32x2 dynamic_extent = {}, size_t max_alloc = SIZE_MAX);
VISP_API birefnet_buffers birefnet_precompute(model_ref, birefnet_params const&);
VISP_API i32x2 birefnet_image_extent(
    i32x2 input_extent, birefnet_params const&, size_t max_alloc = SIZE_MAX);

VISP_API image_data birefnet_process_input(image_view, birefnet_params const&);
VISP_API image_data birefnet_process_output(
    std::span<float const> output_data, i32x2 target_extent, birefnet_params const&);

VISP_API tensor birefnet_predict(model_ref, tensor image, birefnet_params const&);

//
// CLIP vision - image embeddings (IP-Adapter, Wan/Hunyuan image conditioning, unCLIP)

struct clip_model;

struct clip_params {
    int image_size = 224;
    int patch_size = 14;
    int embed_dim = 1280;
    int n_heads = 16;
    int n_layers = 32;
    int projection_dim = 1024;
    bool quick_gelu = false;
    float eps = 1e-5f;
};

struct clip_output {
    tensor image_embeds = nullptr; // [projection_dim, batch]
    tensor penultimate = nullptr;  // [embed_dim, n_patches + 1, batch], second-to-last layer
};

struct clip_result {
    std::vector<float> image_embeds;
    std::vector<float> penultimate;
    int n_tokens = 0;
};

// Loads a CLIP vision model (ViT + projection) from GGUF file onto the backend device.
VISP_API clip_model clip_load_model(char const* filepath, backend_device const&);

// Computes the projected image embedding and the penultimate hidden states.
VISP_API clip_result clip_compute(clip_model&, image_view image);

VISP_API clip_params clip_detect_params(model_file const&);
VISP_API image_data clip_process_input(image_view image, clip_params const&);
VISP_API clip_output clip_encode(model_ref, tensor image, clip_params const&);

//
// YOLOX - object detection (COCO 80 classes for the official weights)

struct yolox_model;

struct yolox_params {
    int input_size = 640;
    int depth = 1;
    int num_classes = 80;
};

struct yolox_output {
    tensor raw = nullptr; // [5 + num_classes, n_cells]: dx, dy, log w, log h, obj, classes
};

struct yolox_box {
    std::array<float, 4> box{}; // x0, y0, x1, y1 in input-image pixels
    float score = 0;
    int label = 0;
};

VISP_API yolox_model yolox_load_model(char const* filepath, backend_device const&);
VISP_API std::vector<yolox_box> yolox_compute(
    yolox_model&, image_view image, float score_thr = 0.3f, float nms_thr = 0.45f);

VISP_API yolox_params yolox_detect_params(model_file const&);
VISP_API image_data yolox_process_input(image_view image, yolox_params const&, float* ratio);
VISP_API yolox_output yolox_detect(model_ref, tensor image, yolox_params const&);
VISP_API std::vector<yolox_box> yolox_decode(
    span<float const> raw, yolox_params const&, float ratio, float score_thr, float nms_thr);

//
// RTMPose / DWPose - top-down whole-body pose estimation (keypoints for one person box)

struct rtmpose_model;

struct rtmpose_params {
    int input_width = 288;
    int input_height = 384;
    int num_keypoints = 133;
    int final_kernel = 7;
    int gau_s = 128;
    int gau_e = 512;
    std::array<int, 4> stage_depths{3, 6, 6, 3};
};

struct rtmpose_output {
    tensor simcc = nullptr; // [2*W + 2*H, keypoints]: SimCC x logits, then y logits
};

struct rtmpose_crop {
    float cx = 0, cy = 0, w = 0, h = 0; // crop centre and size in input-image pixels
};

struct rtmpose_keypoint {
    float x = 0, y = 0, score = 0; // input-image pixels
};

VISP_API rtmpose_model rtmpose_load_model(char const* filepath, backend_device const&);
VISP_API std::vector<rtmpose_keypoint> rtmpose_compute(
    rtmpose_model&, image_view image, std::array<float, 4> box);

VISP_API rtmpose_params rtmpose_detect_params(model_file const&);
VISP_API image_data rtmpose_process_input(
    image_view image, std::array<float, 4> box, rtmpose_params const&, rtmpose_crop* crop);
VISP_API rtmpose_output rtmpose_detect(model_ref, tensor image, rtmpose_params const&);
VISP_API std::vector<rtmpose_keypoint> rtmpose_decode(
    span<float const> simcc, rtmpose_params const&, rtmpose_crop const&);

//
// GFPGAN - blind face restoration of an aligned face crop (v1.3/v1.4 "clean")

struct gfpgan_model;

struct gfpgan_params {
    int image_size = 512;
    int style_dim = 512;
    int num_latent = 16;

    int levels() const { // number of U-Net down/up steps (512 -> 7)
        int n = 0;
        for (int s = image_size; s > 4; s /= 2) ++n;
        return n;
    }
};

// Loads a GFPGAN model; the weights are kept in WHCN layout (bilinear resampling).
VISP_API gfpgan_model gfpgan_load_model(char const* filepath, backend_device const&);

// Restores an aligned face crop (resized to image_size if needed). Returns rgba_u8.
VISP_API image_data gfpgan_compute(gfpgan_model&, image_view face);

VISP_API gfpgan_params gfpgan_detect_params(model_file const&);
VISP_API tensor gfpgan_restore(model_ref, tensor image, gfpgan_params const&);

//
// RIFE - video frame interpolation (Practical-RIFE IFNet v4.2x)

struct rife_model;

struct rife_params {
    int num_blocks = 5;       // flow refinement levels (1/16 .. 1/1)
    int feature_channels = 4; // encoder output channels per frame
    static constexpr int alignment = 64; // input sizes are padded to a multiple of this
};

// Loads a RIFE model; the weights are kept in WHCN layout (bilinear resampling, warping).
VISP_API rife_model rife_load_model(char const* filepath, backend_device const&);

// Synthesizes the frame at time t in (0, 1) between frame0 and frame1 (same size). Returns rgba_u8.
VISP_API image_data rife_compute(rife_model&, image_view frame0, image_view frame1, float t = 0.5f);

VISP_API rife_params rife_detect_params(model_file const&);
// Converts to rgb_f32 and zero-pads to a multiple of rife_params::alignment (bottom/right).
VISP_API image_data rife_process_input(image_view frame);
VISP_API tensor rife_interpolate(model_ref, tensor frame0, tensor frame1, float t, rife_params const&);

//
// Depth Anything - depth estimation

struct depthany_model;

// Loads a Depth Anything V2 model from GGUF file onto the backend device.
// * supports Small/Base/Large variants with flexible input resolution
VISP_API depthany_model depthany_load_model(char const* filepath, backend_device const&);

// Takes RGB input and computes estimated depth (distance from camera).
// Output is a single-channel float32 image in range [0, 1.0].
VISP_API image_data depthany_compute(depthany_model&, image_view image);

// --- Depth Anything pipeline

struct depthany_params {
    int image_size = 518;
    int image_multiple = 14;
    i32x2 image_extent = {518, 518};
    float max_depth = 1;
    std::array<int, 4> feature_layers = {2, 5, 8, 11};
    dino_params dino;
};

VISP_API depthany_params depthany_detect_params(model_file const&, i32x2 input_extent = {});
VISP_API i32x2 depthany_image_extent(i32x2 input_extent, depthany_params const&);

VISP_API image_data depthany_process_input(image_view image, depthany_params const&);
VISP_API image_data depthany_process_output(
    std::span<float const> output_data, i32x2 target_extent, depthany_params const&);

VISP_API tensor depthany_predict(model_ref, tensor image, depthany_params const&);

//
// MI-GAN - image inpainting

struct migan_model;

// Loads a MI-GAN model from GGUF file onto the backend device.
// * supports variants at 256px or 512px resolution
VISP_API migan_model migan_load_model(char const* filepath, backend_device const&);

// Fills pixels in the input image where the mask is 1.0 with new content.
VISP_API image_data migan_compute(migan_model&, image_view image, image_view mask);

// --- MI-GAN pipeline

struct migan_params {
    int resolution = 256;
    bool invert_mask = false;
};

VISP_API migan_params migan_detect_params(model_file const&);

VISP_API image_data migan_process_input(image_view image, image_view mask, migan_params const&);
VISP_API image_data migan_process_output(
    std::span<float const> data, i32x2 extent, migan_params const&);

VISP_API tensor migan_generate(model_ref, tensor image, migan_params const&);

//
// ESRGAN - image super-resolution

struct esrgan_model;

// Loads an ESRGAN model from GGUF file onto the backend device.
// * supports ESRGAN, RealESRGAN variants with flexible scale and number of blocks
// * currently does not spport RealESRGAN+ (plus) models or those which use pixel shuffle
VISP_API esrgan_model esrgan_load_model(char const* filepath, backend_device const&);

// Upscales the input image by the model's scale factor. Uses tiling for large inputs.
VISP_API image_data esrgan_compute(esrgan_model&, image_view image);

// --- ESRGAN pipeline

struct esrgan_params {
    int scale = 4;
    int n_blocks = 23;
};

VISP_API esrgan_params esrgan_detect_params(model_file const&);
VISP_API int esrgan_estimate_graph_size(esrgan_params const&);

VISP_API tensor esrgan_generate(model_ref, tensor image, esrgan_params const&);

//
// Implementation

// internal
struct sam_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    sam_params params;

    compute_graph encoder;
    i32x2 image_extent{};
    tensor input_image = nullptr;
    tensor output_embed = nullptr;

    compute_graph decoder;
    tensor input_embed = nullptr;
    tensor input_prompt = nullptr;
    sam_prediction output = {};
    bool is_point_prompt = true;
};

// internal
struct birefnet_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    birefnet_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct yolox_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    yolox_params params;

    compute_graph graph;
    tensor input = nullptr;
    yolox_output output;
};

// internal
struct rtmpose_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    rtmpose_params params;

    compute_graph graph;
    tensor input = nullptr;
    rtmpose_output output;
};

// internal
struct gfpgan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    gfpgan_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct rife_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    rife_params params;

    compute_graph graph;
    i32x2 extent{};
    float t = -1.f;
    tensor input0 = nullptr;
    tensor input1 = nullptr;
    tensor output = nullptr;
};

// internal
struct clip_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    clip_params params;

    compute_graph graph;
    tensor input = nullptr;
    clip_output output;
};

// internal
struct depthany_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    depthany_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct migan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    migan_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct esrgan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    esrgan_params params;

    compute_graph graph;
    i32x2 tile_size{};
    tensor input = nullptr;
    tensor output = nullptr;
};

} // namespace visp