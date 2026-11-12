//! visp-convert: turn `.safetensors` weights into vision.cpp GGUF files.
//!
//! usage: visp-convert <arch> <input.safetensors> <output.gguf> [--f32] [--quick-gelu]
//! arch: clip, yolox (use `pickle2safetensors --key model` on the official .pth first)

mod gguf;

use gguf::{DType, Tensor, Value, Writer};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// One source tensor decoded to f32, with its PyTorch shape.
struct Source {
    shape: Vec<u64>,
    values: Vec<f32>,
}

fn decode(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let bytes = view.data();
    let out = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(format!("unsupported dtype {other:?}").into()),
    };
    Ok(out)
}

fn load(path: &Path) -> Result<BTreeMap<String, Source>> {
    let file = File::open(path)?;
    // SAFETY: the file is opened read-only and not modified while mapped.
    #[allow(unsafe_code)]
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&map)?;
    let mut out = BTreeMap::new();
    for (name, view) in st.tensors() {
        if !matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            continue; // e.g. int64 position_ids
        }
        let shape = view.shape().iter().map(|&d| d as u64).collect();
        out.insert(name, Source { shape, values: decode(&view)? });
    }
    Ok(out)
}

/// Permute a conv kernel from PyTorch [O, I, H, W] to [O, H, W, I] (vision.cpp CWHN).
fn conv_to_cwhn(src: &Source) -> Result<Source> {
    let [o, i, h, w] = <[u64; 4]>::try_from(src.shape.as_slice())
        .map_err(|_| "conv kernel must be 4-D")?
        .map(|d| d as usize);
    let mut values = vec![0f32; src.values.len()];
    for oo in 0..o {
        for ii in 0..i {
            for y in 0..h {
                for x in 0..w {
                    let from = ((oo * i + ii) * h + y) * w + x;
                    let to = ((oo * h + y) * w + x) * i + ii;
                    if let (Some(dst), Some(v)) = (values.get_mut(to), src.values.get(from)) {
                        *dst = *v;
                    }
                }
            }
        }
    }
    Ok(Source { shape: [o, h, w, i].map(|d| d as u64).to_vec(), values })
}

fn get<'a>(m: &'a BTreeMap<String, Source>, key: &str) -> Result<&'a Source> {
    m.get(key).ok_or_else(|| format!("missing tensor {key}").into())
}

fn convert_clip(model: &BTreeMap<String, Source>, g: &mut Writer, f16: bool, quick: bool) -> Result<()> {
    let p = "vision_model.";
    let patch = get(model, &format!("{p}embeddings.patch_embedding.weight"))?;
    let pos = get(model, &format!("{p}embeddings.position_embedding.weight"))?;
    let proj = get(model, "visual_projection.weight")?;
    let embed_dim = *patch.shape.first().ok_or("bad patch shape")?;
    let patch_size = *patch.shape.get(2).ok_or("bad patch shape")?;
    let n_pos = *pos.shape.first().ok_or("bad pos shape")?;
    let grid = ((n_pos - 1) as f64).sqrt().round() as u64;
    let n_layers = model
        .keys()
        .filter_map(|k| k.strip_prefix(&format!("{p}encoder.layers.")))
        .filter_map(|k| k.split('.').next()?.parse::<u64>().ok())
        .max()
        .ok_or("no encoder layers")?
        + 1;
    let projection_dim = *proj.shape.first().ok_or("bad projection shape")?;

    let int = |v: u64| Value::I32(i32::try_from(v).unwrap_or(i32::MAX));
    g.kv("general.architecture", Value::Str("clip".into()));
    g.kv("general.file_type", Value::U32(u32::from(f16)));
    g.kv("clip.tensor_data_layout", Value::Str("whcn".into()));
    g.kv("clip.image_size", int(grid * patch_size));
    g.kv("clip.patch_size", int(patch_size));
    g.kv("clip.embed_dim", int(embed_dim));
    g.kv("clip.n_heads", Value::I32(16));
    g.kv("clip.n_layers", int(n_layers));
    g.kv("clip.projection_dim", int(projection_dim));
    g.kv("clip.quick_gelu", Value::I32(i32::from(quick)));
    println!(
        "clip: image {} patch {patch_size} dim {embed_dim} layers {n_layers} proj {projection_dim} quick_gelu {quick}",
        grid * patch_size
    );

    for (name, src) in model {
        let short = name.strip_prefix(p).unwrap_or(name);
        if short.contains("position_ids") {
            continue;
        }
        let converted;
        let src = if short == "embeddings.patch_embedding.weight" {
            converted = conv_to_cwhn(src)?;
            &converted
        } else {
            src
        };
        // Vectors, embeddings and norms stay f32; matrices follow the requested type.
        let keep_f32 = !f16 || src.shape.len() < 2 || short.starts_with("embeddings.");
        let dtype = if keep_f32 { DType::F32 } else { DType::F16 };
        g.tensor(Tensor::from_f32(short, src.shape.clone(), &src.values, dtype))?;
    }
    Ok(())
}

/// Rewrite YOLOX's Focus stem (space-to-depth + 3x3 conv) as an equivalent 6x6 stride-2 conv
/// with padding 2. Focus channel groups are (top-left, bottom-left, top-right, bottom-right).
fn focus_to_conv6(src: &Source) -> Result<Source> {
    let [o, i4, kh, kw] = <[u64; 4]>::try_from(src.shape.as_slice())
        .map_err(|_| "stem kernel must be 4-D")?
        .map(|d| d as usize);
    if kh != 3 || kw != 3 || i4 % 4 != 0 {
        return Err("unexpected Focus stem kernel".into());
    }
    let c = i4 / 4;
    let mut values = vec![0f32; o * c * 36];
    for oo in 0..o {
        for (g, (dy, dx)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            for cc in 0..c {
                for a in 0..3 {
                    for b in 0..3 {
                        let from = ((oo * i4 + g * c + cc) * 3 + a) * 3 + b;
                        let to = ((oo * c + cc) * 6 + 2 * a + dy) * 6 + 2 * b + dx;
                        if let (Some(d), Some(v)) = (values.get_mut(to), src.values.get(from)) {
                            *d = *v;
                        }
                    }
                }
            }
        }
    }
    Ok(Source { shape: [o, c, 6, 6].map(|d| d as u64).to_vec(), values })
}

fn convert_yolox(model: &BTreeMap<String, Source>, g: &mut Writer, f16: bool) -> Result<()> {
    const EPS: f32 = 1e-3; // YOLOX sets BatchNorm eps to 1e-3 (init_yolo)
    let stem = get(model, "backbone.backbone.stem.conv.conv.weight")?;
    let base_channels = *stem.shape.first().ok_or("bad stem")?;
    let depth = model
        .keys()
        .filter_map(|k| k.strip_prefix("backbone.backbone.dark2.1.m."))
        .filter_map(|k| k.split('.').next()?.parse::<u64>().ok())
        .max()
        .ok_or("no dark2 bottlenecks")?
        + 1;
    let classes = *get(model, "head.cls_preds.0.bias")?.shape.first().ok_or("bad cls")?;
    if model.keys().any(|k| k.contains(".dconv.")) {
        return Err("depthwise YOLOX (nano/tiny) is not supported".into());
    }
    let int = |v: u64| Value::I32(i32::try_from(v).unwrap_or(i32::MAX));
    g.kv("general.architecture", Value::Str("yolox".into()));
    g.kv("general.file_type", Value::U32(u32::from(f16)));
    g.kv("yolox.tensor_data_layout", Value::Str("whcn".into()));
    g.kv("yolox.base_channels", int(base_channels));
    g.kv("yolox.depth", int(depth));
    g.kv("yolox.num_classes", int(classes));
    println!("yolox: base channels {base_channels}, depth {depth}, classes {classes}");

    let mut conv2d = Vec::new();
    let mut push = |g: &mut Writer, name: &str, shape: Vec<u64>, values: &[f32]| -> Result<()> {
        let is_conv = shape.len() == 4;
        if is_conv {
            conv2d.push(i32::try_from(g.len())?);
        }
        let dtype = if is_conv && f16 { DType::F16 } else { DType::F32 };
        g.tensor(Tensor::from_f32(name, shape, values, dtype))?;
        Ok(())
    };
    for (name, src) in model {
        let Some(prefix) = name.strip_suffix(".conv.weight") else {
            if (name.contains("_preds.")) && (name.ends_with(".weight") || name.ends_with(".bias")) {
                push(g, name, src.shape.clone(), &src.values)?;
            }
            continue;
        };
        let stem_conv;
        let w = if prefix == "backbone.backbone.stem.conv" {
            stem_conv = focus_to_conv6(src)?;
            &stem_conv
        } else {
            src
        };
        let bn = |s: &str| get(model, &format!("{prefix}.bn.{s}"));
        let (gamma, beta, mean, var) = (bn("weight")?, bn("bias")?, bn("running_mean")?, bn("running_var")?);
        let out = gamma.values.len();
        let per = w.values.len() / out;
        let mut weight = w.values.clone();
        let mut bias = vec![0f32; out];
        for o in 0..out {
            let (Some(ga), Some(be), Some(mu), Some(va)) =
                (gamma.values.get(o), beta.values.get(o), mean.values.get(o), var.values.get(o))
            else {
                return Err(format!("{prefix}: bn size mismatch").into());
            };
            let k = ga / (va + EPS).sqrt();
            for v in weight.iter_mut().skip(o * per).take(per) {
                *v *= k;
            }
            if let Some(b) = bias.get_mut(o) {
                *b = be - mu * k;
            }
        }
        push(g, &format!("{prefix}.conv.weight"), w.shape.clone(), &weight)?;
        push(g, &format!("{prefix}.conv.bias"), vec![out as u64], &bias)?;
    }
    g.kv("yolox.conv2d_weights", Value::I32Array(conv2d));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [arch, input, output] = pos.as_slice() else {
        return Err("usage: visp-convert <clip|yolox> <input.safetensors> <output.gguf> [--f32] [--quick-gelu]".into());
    };
    let model = load(Path::new(input.as_str()))?;
    let mut g = Writer::default();
    match arch.as_str() {
        "clip" => convert_clip(&model, &mut g, !flag("--f32"), flag("--quick-gelu"))?,
        "yolox" => convert_yolox(&model, &mut g, !flag("--f32"))?,
        other => return Err(format!("unknown arch {other}").into()),
    }
    let size = g.write(Path::new(output.as_str()))?;
    println!("wrote {} tensors, {size} bytes -> {output}", g.len());
    Ok(())
}
