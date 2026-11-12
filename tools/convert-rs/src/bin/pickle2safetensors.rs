//! Convert a PyTorch zip checkpoint (`torch.save` of a state dict, e.g. `photomaker-v2.bin`)
//! into `.safetensors` without Python or PyTorch.
//!
//! usage: pickle2safetensors <input.bin|.pt|.pth> <output.safetensors>
//!
//! The pickle is *interpreted as data*, never executed: only the opcodes a state dict needs are
//! implemented, and only these globals are accepted: `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2` and `torch.{Float,Half,BFloat16,Long}Storage`. Anything else
//! (arbitrary callables, `os.system`, ...) is rejected before any data is touched.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    List(Vec<Val>),
    Str(String),
    Tuple(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String),
    /// An object of a class outside the allow-list (e.g. numpy metadata). Never constructed or
    /// called; kept only as an inert placeholder and skipped on output.
    Opaque(String),
    Storage { dtype: Dtype, key: String },
    Tensor(Tensor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    F32,
    F16,
    Bf16,
    I64,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::Bf16 => 2,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I64 => "I64",
        }
    }
}

#[derive(Clone, Debug)]
struct Tensor {
    dtype: Dtype,
    key: String,
    offset: usize,
    shape: Vec<usize>,
}

const ALLOWED: [&str; 6] = [
    "torch LongStorage",
    "collections OrderedDict",
    "torch._utils _rebuild_tensor_v2",
    "torch FloatStorage",
    "torch HalfStorage",
    "torch BFloat16Storage",
];

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated pickle")?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(*self.take(1)?.first().ok_or("truncated pickle")?)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn line(&mut self) -> Result<String> {
        let rest = self.b.get(self.p..).ok_or("truncated pickle")?;
        let n = rest.iter().position(|&c| c == b'\n').ok_or("unterminated GLOBAL")?;
        let s = String::from_utf8(self.take(n)?.to_vec())?;
        self.p += 1;
        Ok(s)
    }
}

fn int(v: &Val) -> Result<usize> {
    match v {
        Val::Int(i) => Ok(usize::try_from(*i)?),
        _ => Err(format!("expected int, got {v:?}").into()),
    }
}

fn ints(v: &Val) -> Result<Vec<usize>> {
    match v {
        Val::Tuple(t) => t.iter().map(int).collect(),
        _ => Err(format!("expected tuple, got {v:?}").into()),
    }
}

fn storage(pid: Val) -> Result<Val> {
    let Val::Tuple(t) = pid else { return Err("persistent id is not a tuple".into()) };
    let (Some(Val::Str(kind)), Some(Val::Global(ty)), Some(Val::Str(key))) = (t.first(), t.get(1), t.get(2)) else {
        return Err(format!("unexpected persistent id {t:?}").into());
    };
    if kind != "storage" {
        return Err(format!("unexpected persistent id kind {kind}").into());
    }
    let dtype = match ty.as_str() {
        "torch FloatStorage" => Dtype::F32,
        "torch HalfStorage" => Dtype::F16,
        "torch BFloat16Storage" => Dtype::Bf16,
        "torch LongStorage" => Dtype::I64,
        other => return Err(format!("unsupported storage {other}").into()),
    };
    Ok(Val::Storage { dtype, key: key.clone() })
}

fn reduce(f: Val, args: Val) -> Result<Val> {
    let name = match f {
        Val::Global(name) => name,
        Val::Opaque(name) => return Ok(Val::Opaque(name)),
        _ => return Err("REDUCE on a non-global".into()),
    };
    let Val::Tuple(a) = args else { return Err("REDUCE args are not a tuple".into()) };
    match name.as_str() {
        "collections OrderedDict" => Ok(Val::Dict(Vec::new())),
        "torch._utils _rebuild_tensor_v2" => {
            let (Some(Val::Storage { dtype, key }), Some(off), Some(size), Some(stride)) =
                (a.first(), a.get(1), a.get(2), a.get(3))
            else {
                return Err("bad _rebuild_tensor_v2 arguments".into());
            };
            let shape = ints(size)?;
            let stride = ints(stride)?;
            let mut want = 1;
            for (d, s) in shape.iter().zip(&stride).rev() {
                if *d > 1 && *s != want {
                    return Err(format!("non-contiguous tensor (shape {shape:?}, stride {stride:?})").into());
                }
                want *= d;
            }
            Ok(Val::Tensor(Tensor { dtype: *dtype, key: key.clone(), offset: int(off)?, shape }))
        }
        other => Err(format!("refusing to call {other}").into()),
    }
}

fn unpickle(b: &[u8]) -> Result<Val> {
    let mut r = Reader { b, p: 0 };
    let mut stack: Vec<Val> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let pop = |s: &mut Vec<Val>| s.pop().ok_or("pickle stack underflow");
    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?; // PROTO
            }
            b'.' => return pop(&mut stack).map_err(Into::into), // STOP
            b'}' => stack.push(Val::Dict(Vec::new())),
            b']' => stack.push(Val::List(Vec::new())),
            b'G' => stack.push(Val::Float(f64::from_be_bytes(r.take(8)?.try_into()?))),
            b'e' | b'a' => {
                let items = if op == b'a' {
                    vec![pop(&mut stack)?]
                } else {
                    let m = marks.pop().ok_or("APPENDS without MARK")?;
                    stack.split_off(m)
                };
                if matches!(stack.last(), Some(Val::Opaque(_))) {
                    continue;
                }
                let Some(Val::List(l)) = stack.last_mut() else { return Err("APPEND(S) on a non-list".into()) };
                l.extend(items);
            }
            b'b' => {
                // BUILD: object state (e.g. a state dict's `_metadata`); the data itself is kept.
                pop(&mut stack)?;
            }
            b')' => stack.push(Val::Tuple(Vec::new())),
            b'N' => stack.push(Val::None),
            0x88 => stack.push(Val::Bool(true)),
            0x89 => stack.push(Val::Bool(false)),
            b'(' => marks.push(stack.len()),
            b'J' => stack.push(Val::Int(i64::from(i32::from_le_bytes(r.take(4)?.try_into()?)))),
            b'K' => stack.push(Val::Int(i64::from(r.u8()?))),
            b'M' => stack.push(Val::Int(i64::from(r.u16()?))),
            b'X' => {
                let n = r.u32()? as usize;
                stack.push(Val::Str(String::from_utf8(r.take(n)?.to_vec())?));
            }
            0x8c => {
                let n = r.u8()? as usize;
                stack.push(Val::Str(String::from_utf8(r.take(n)?.to_vec())?));
            }
            b'q' => {
                let k = u32::from(r.u8()?);
                memo.insert(k, stack.last().ok_or("BINPUT on empty stack")?.clone());
            }
            b'r' => {
                let k = r.u32()?;
                memo.insert(k, stack.last().ok_or("LONG_BINPUT on empty stack")?.clone());
            }
            b'h' => {
                let k = u32::from(r.u8()?);
                stack.push(memo.get(&k).ok_or("BINGET of unknown memo")?.clone());
            }
            b'j' => {
                let k = r.u32()?;
                stack.push(memo.get(&k).ok_or("LONG_BINGET of unknown memo")?.clone());
            }
            b'c' => {
                let g = format!("{} {}", r.line()?, r.line()?);
                if ALLOWED.contains(&g.as_str()) {
                    stack.push(Val::Global(g));
                } else {
                    stack.push(Val::Opaque(g));
                }
            }
            b't' => {
                let m = marks.pop().ok_or("TUPLE without MARK")?;
                let items = stack.split_off(m);
                stack.push(Val::Tuple(items));
            }
            0x85..=0x87 => {
                let n = usize::from(op - 0x84);
                let at = stack.len().checked_sub(n).ok_or("pickle stack underflow")?;
                let items = stack.split_off(at);
                stack.push(Val::Tuple(items));
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(storage(pid)?);
            }
            0x81 => {
                // NEWOBJ cls args: only ever applied to non-allow-listed classes -> placeholder.
                let _args = pop(&mut stack)?;
                let cls = pop(&mut stack)?;
                let name = match cls {
                    Val::Opaque(n) | Val::Global(n) => n,
                    _ => "?".into(),
                };
                stack.push(Val::Opaque(name));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                stack.push(reduce(f, args)?);
            }
            b's' | b'u' => {
                let items = if op == b's' {
                    let v = pop(&mut stack)?;
                    let k = pop(&mut stack)?;
                    vec![k, v]
                } else {
                    let m = marks.pop().ok_or("SETITEMS without MARK")?;
                    stack.split_off(m)
                };
                if matches!(stack.last(), Some(Val::Opaque(_))) {
                    continue;
                }
                let Some(Val::Dict(d)) = stack.last_mut() else { return Err("SETITEM(S) on a non-dict".into()) };
                for kv in items.chunks(2) {
                    if let [k, v] = kv {
                        d.push((k.clone(), v.clone()));
                    }
                }
            }
            other => return Err(format!("unsupported pickle opcode 0x{other:02x} at {}", r.p - 1).into()),
        }
    }
}

fn flatten(prefix: &str, v: Val, out: &mut Vec<(String, Tensor)>) -> Result<()> {
    match v {
        Val::Dict(items) => {
            for (k, v) in items {
                let Val::Str(k) = k else { return Err(format!("non-string key {k:?}").into()) };
                let name = if prefix.is_empty() { k } else { format!("{prefix}.{k}") };
                flatten(&name, v, out)?;
            }
            Ok(())
        }
        Val::Tensor(t) => {
            out.push((prefix.to_string(), t));
            Ok(())
        }
        Val::None | Val::Bool(_) | Val::Int(_) | Val::Float(_) | Val::Str(_) | Val::List(_) | Val::Opaque(_) => {
            eprintln!("skipping non-tensor entry {prefix}");
            Ok(())
        }
        other => Err(format!("{prefix}: unsupported value {other:?}").into()),
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let photomaker = args.iter().any(|a| a == "--photomaker");
    let key = args.iter().position(|a| a == "--key").and_then(|i| args.get(i + 1)).cloned();
    let pos: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && !(*i > 0 && args.get(i - 1).is_some_and(|p| p == "--key")))
        .map(|(_, a)| a)
        .collect();
    let [input, output] = pos.as_slice() else {
        return Err("usage: pickle2safetensors <input.bin> <output.safetensors> [--photomaker] [--key NAME]".into());
    };
    let mut zip = zip::ZipArchive::new(File::open(input)?)?;
    let pkl = (0..zip.len())
        .filter_map(|i| zip.name_for_index(i).map(str::to_string))
        .find(|n| n.ends_with("/data.pkl") || n == "data.pkl")
        .ok_or("no data.pkl: not a torch zip checkpoint (legacy pickles are not supported)")?;
    let root = pkl.strip_suffix("data.pkl").unwrap_or_default().to_string();
    let mut bytes = Vec::new();
    zip.by_name(&pkl)?.read_to_end(&mut bytes)?;

    let mut tensors = Vec::new();
    let mut root_val = unpickle(&bytes)?;
    if let Some(k) = &key {
        let Val::Dict(items) = root_val else { return Err("--key needs a dict at the root".into()) };
        root_val = items
            .into_iter()
            .find(|(n, _)| matches!(n, Val::Str(s) if s == k))
            .map(|(_, v)| v)
            .ok_or_else(|| format!("no top-level key {k}"))?;
    }
    flatten("", root_val, &mut tensors)?;
    if photomaker {
        // stable-diffusion.cpp's PhotoMaker loader expects the layout of the community
        // conversion: no `id_encoder.`/`lora_weights.` level, HF's `pre_layrnorm` typo fixed,
        // and the projection under `vision_model.`.
        for (name, _) in &mut tensors {
            let mut n = name
                .strip_prefix("id_encoder.")
                .or_else(|| name.strip_prefix("lora_weights."))
                .unwrap_or(name)
                .replace("pre_layrnorm", "pre_layernorm");
            if n == "visual_projection.weight" {
                n = "vision_model.visual_projection.weight".into();
            }
            *name = n;
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    if tensors.windows(2).any(|w| matches!(w, [a, b] if a.0 == b.0)) {
        return Err("duplicate tensor names after renaming".into());
    }

    let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut end = 0usize;
    for (name, t) in &tensors {
        let n: usize = t.shape.iter().product();
        let start = end;
        end += n * t.dtype.size();
        let shape: Vec<String> = t.shape.iter().map(ToString::to_string).collect();
        header.push_str(&format!(
            ",{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
            json_str(name),
            t.dtype.name(),
            shape.join(",")
        ));
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }

    let mut w = BufWriter::new(File::create(output)?);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    let mut cache: Option<(String, Vec<u8>)> = None;
    for (name, t) in &tensors {
        if cache.as_ref().map(|c| &c.0) != Some(&t.key) {
            let mut data = Vec::new();
            zip.by_name(&format!("{root}data/{}", t.key))?.read_to_end(&mut data)?;
            cache = Some((t.key.clone(), data));
        }
        let data = cache.as_ref().map(|c| &c.1).ok_or("storage cache")?;
        let es = t.dtype.size();
        let n: usize = t.shape.iter().product();
        let slice = data
            .get(t.offset * es..(t.offset + n) * es)
            .ok_or_else(|| format!("{name}: storage {} too small", t.key))?;
        w.write_all(slice)?;
    }
    w.flush()?;
    eprintln!("{} tensors, {end} data bytes -> {output}", tensors.len());
    Ok(())
}
