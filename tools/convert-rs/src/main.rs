//! visp-convert: turn `.safetensors` weights into vision.cpp GGUF files.
//!
//! usage: visp-convert <arch> <input.safetensors> <output.gguf> [--f32] [--quick-gelu]
//! arch: clip

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [arch, input, output] = pos.as_slice() else {
        return Err("usage: visp-convert <clip> <input.safetensors> <output.gguf> [--f32] [--quick-gelu]".into());
    };
    let model = load(Path::new(input.as_str()))?;
    let mut g = Writer::default();
    match arch.as_str() {
        "clip" => convert_clip(&model, &mut g, !flag("--f32"), flag("--quick-gelu"))?,
        other => return Err(format!("unknown arch {other}").into()),
    }
    let size = g.write(Path::new(output.as_str()))?;
    println!("wrote {} tensors, {size} bytes -> {output}", g.len());
    Ok(())
}
