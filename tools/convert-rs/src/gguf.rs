//! Minimal GGUF v3 writer: metadata key/values plus F32/F16 tensors.
//!
//! Dimensions are given in PyTorch order (outermost first) and written in ggml
//! order (innermost first), so callers never reverse shapes by hand.

use std::fs::File;
use std::io::{self, BufWriter, Seek, Write};
use std::path::Path;

const ALIGNMENT: u64 = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
        }
    }
    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

pub enum Value {
    U32(u32),
    I32(i32),
    F32(f32),
    Str(String),
    I32Array(Vec<i32>),
}

pub struct Tensor {
    pub name: String,
    /// PyTorch order, outermost dimension first.
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl Tensor {
    /// Build a tensor from f32 values, storing them as `dtype`.
    pub fn from_f32(name: &str, shape: Vec<u64>, values: &[f32], dtype: DType) -> Self {
        let data = match dtype {
            DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect(),
        };
        Self { name: name.to_owned(), shape, dtype, data }
    }
}

#[derive(Default)]
pub struct Writer {
    kv: Vec<(String, Value)>,
    tensors: Vec<Tensor>,
}

fn put_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn padding(n: u64) -> u64 {
    (ALIGNMENT - n % ALIGNMENT) % ALIGNMENT
}

fn zeros(n: u64) -> Vec<u8> {
    vec![0u8; usize::try_from(n).unwrap_or(0)]
}

impl Writer {
    pub fn kv(&mut self, key: &str, value: Value) {
        self.kv.push((key.to_owned(), value));
    }

    pub fn tensor(&mut self, t: Tensor) -> io::Result<()> {
        if t.name.len() >= 64 {
            return Err(io::Error::other(format!("tensor name too long: {}", t.name)));
        }
        let elements: u64 = t.shape.iter().product();
        if elements * t.dtype.size() as u64 != t.data.len() as u64 {
            return Err(io::Error::other(format!("{}: data size does not match shape", t.name)));
        }
        self.tensors.push(t);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn write(&self, path: &Path) -> io::Result<u64> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(self.kv.len() as u64).to_le_bytes())?;

        for (key, value) in &self.kv {
            put_str(&mut w, key)?;
            match value {
                Value::U32(v) => {
                    w.write_all(&4u32.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                Value::I32(v) => {
                    w.write_all(&5u32.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                Value::F32(v) => {
                    w.write_all(&6u32.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                Value::Str(s) => {
                    w.write_all(&8u32.to_le_bytes())?;
                    put_str(&mut w, s)?;
                }
                Value::I32Array(a) => {
                    w.write_all(&9u32.to_le_bytes())?;
                    w.write_all(&5u32.to_le_bytes())?;
                    w.write_all(&(a.len() as u64).to_le_bytes())?;
                    for v in a {
                        w.write_all(&v.to_le_bytes())?;
                    }
                }
            }
        }

        let mut offset = 0u64;
        for t in &self.tensors {
            put_str(&mut w, &t.name)?;
            w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
            for d in t.shape.iter().rev() {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&t.dtype.ggml_type().to_le_bytes())?;
            w.write_all(&offset.to_le_bytes())?;
            let n = t.data.len() as u64;
            offset += n + padding(n);
        }

        let header = w.stream_position()?;
        w.write_all(&zeros(padding(header)))?;
        for t in &self.tensors {
            w.write_all(&t.data)?;
            w.write_all(&zeros(padding(t.data.len() as u64)))?;
        }
        w.flush()?;
        w.stream_position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_alignment() -> io::Result<()> {
        let path = std::env::temp_dir().join("visp-convert-test.gguf");
        let mut g = Writer::default();
        g.kv("general.architecture", Value::Str("test".into()));
        g.tensor(Tensor::from_f32("a", vec![2, 3], &[1.0; 6], DType::F16))?;
        g.tensor(Tensor::from_f32("b", vec![5], &[2.0; 5], DType::F32))?;
        let size = g.write(&path)?;
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes.len() as u64, size);
        assert_eq!(bytes.get(0..4), Some(&b"GGUF"[..]));
        assert_eq!(size % ALIGNMENT, 0);
        std::fs::remove_file(path)
    }
}
