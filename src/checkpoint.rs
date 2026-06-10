//! Minimal checkpoint format: `[u32 n] n × { u32 name_len, name utf8,
//! u32 elem_count, f32-LE data }`, plus a trailing meta map of the same
//! shape for scalars (step counter etc).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

pub type Params = BTreeMap<String, Vec<f32>>;

pub fn save(path: &Path, params: &Params, step: u64) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(b"VWCK0001")?;
        f.write_all(&(step).to_le_bytes())?;
        f.write_all(&(params.len() as u32).to_le_bytes())?;
        for (name, data) in params {
            f.write_all(&(name.len() as u32).to_le_bytes())?;
            f.write_all(name.as_bytes())?;
            f.write_all(&(data.len() as u32).to_le_bytes())?;
            for &v in data {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    std::fs::rename(tmp, path)
}

pub fn load(path: &Path) -> std::io::Result<(Params, u64)> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    assert_eq!(&magic, b"VWCK0001", "bad checkpoint magic");
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8)?;
    let step = u64::from_le_bytes(b8);
    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    let n = u32::from_le_bytes(b4);
    let mut params = Params::new();
    for _ in 0..n {
        f.read_exact(&mut b4)?;
        let mut name = vec![0u8; u32::from_le_bytes(b4) as usize];
        f.read_exact(&mut name)?;
        f.read_exact(&mut b4)?;
        let count = u32::from_le_bytes(b4) as usize;
        let mut bytes = vec![0u8; count * 4];
        f.read_exact(&mut bytes)?;
        let data = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        params.insert(String::from_utf8(name).unwrap(), data);
    }
    Ok((params, step))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut params = Params::new();
        params.insert("a.weight".into(), vec![1.0, -2.5, 3.25]);
        params.insert("b.bias".into(), vec![0.0; 7]);
        let dir = std::env::temp_dir().join("vw_ckpt_test.bin");
        save(&dir, &params, 1234).unwrap();
        let (back, step) = load(&dir).unwrap();
        assert_eq!(step, 1234);
        assert_eq!(params, back);
        std::fs::remove_file(dir).ok();
    }
}
