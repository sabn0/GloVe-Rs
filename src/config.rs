

use ndarray::Array2;
use ndarray_npy::{ReadNpyError, read_npy, write_npy};
use serde::{Serialize, Deserialize, Serializer, Deserializer, ser::SerializeSeq, de::Visitor};
use serde_json::Value;
use std::{fs::{self, File}, error::Error, collections::HashMap, fmt::Display, io::{BufWriter, BufReader, IoSlice}};
use std::io::prelude::*;
use flate2::{Compression, read::{GzDecoder}};
use flate2::write::GzEncoder;

#[derive(Clone, Debug)]
pub struct JsonTrain {
    pub vocab_size: usize,
    pub max_iter: usize,
    pub embedding_dim: usize,
    pub learning_rate: f32,
    pub x_max: f32,
    pub alpha: f32,
    pub batch_size: usize,
    pub num_threads_training: usize
}


impl Display for JsonTrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "training hyper parameters:
        vocab_size: {},
        max_iter: {},
        embedding_dim: {},
        learning_rate: {},
        x_max: {},
        alpha: {},
        batch_size: {},
        num_threads_training: {}",
        self.vocab_size, self.max_iter, self.embedding_dim, self.learning_rate, self.x_max, self.alpha, self.batch_size, self.num_threads_training
        )
    }
}

#[derive(Clone, Debug)]
pub struct JsonTypes {
    pub corpus_file: String,
    pub output_dir: String,
    pub window_size: i32,
    pub saved_counts: Option<bool>,
    pub num_threads_cooc: usize,
    pub json_train: JsonTrain
}



impl Display for JsonTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "using hyper-params:
        corpus_file: {}
        output_dir: {}
        window_size: {}
        saved_counts: {:?}
        num_threads_cooc: {},
        Using training hyper-params: {}",
        self.corpus_file, self.output_dir, self.window_size, self.saved_counts, self.num_threads_cooc, self.json_train)
    }
}

pub struct Config {
    params: JsonTypes
}

impl Config {

    pub fn get_params(&self) -> JsonTypes {
        return self.params.clone()
    }

    pub fn new(args: &[String]) -> Result<Config, Box<dyn Error>> { 

        if args.len() != 2 {
            return Err(format!("input should be a path to json file only").into());
        }

        // parse input json
        let f = fs::File::open(&args[1]).expect("cannot open json file");
        let json: Value = serde_json::from_reader(f).expect("cannot read json file");

        // validate input and output in json
        let corpus_file = json.get("corpus_file").expect("corpus_file was not supplied throught json").as_str().expect("cannot cast input file to string");
        let output_dir = json.get("output_dir").expect("output_dir was not supplied throught json").as_str().expect("cannot cast output path to string");

        // handle default vs input parameters
        let vocab_size = match json.get("vocab_size") {
            Some(vocab_size) => vocab_size.as_i64().expect("panic since given vocab_size is not numeric"),
            None => 400000
        };
        let window_size = match json.get("window_size") {
            Some(window_size) => window_size.as_i64().expect("panic since given window_size is not numeric"),
            None => 10
        };
        let learning_rate = match json.get("learning_rate") {
            Some(learning_rate) => learning_rate.as_f64().expect("panic since given learning_rate is not numeric"),
            None => 0.05
        };
        let max_iter = match json.get("max_iter") {
            Some(max_iter) => max_iter.as_i64().expect("panic since given max_iter is not numeric"),
            None => 50
        };
        let embedding_dim = match json.get("embedding_dim") {
            Some(embedding_dim) => embedding_dim.as_i64().expect("panic since given embedding_dim is not numeric"),
            None => 300
        };
        let x_max = match json.get("x_max") {
            Some(x_max) => x_max.as_f64().expect("panic since given x_max is not numeric"),
            None => 100 as f64
        };
        let alpha = match json.get("alpha") {
            Some(alpha) => alpha.as_f64().expect("panic since given alpha is not numeric"),
            None => 0.75
        };
        let batch_size = match json.get("batch_size") {
            Some(batch_size) => batch_size.as_i64().expect("panic since given batch_size is not numeric"),
            None => 1
        };
        let saved_counts = match json.get("saved_counts") {
            Some(saved_counts) => Some(saved_counts.as_bool().expect("panic since given saved_counts is not boolean")),
            None => None
        };
        let num_threads_cooc = match json.get("num_threads_cooc") {
            Some(num_threads_cooc) => num_threads_cooc.as_i64().expect("panic since given num_threads_cooc is not numeric"),
            None => 4
        };
        let num_threads_training = match json.get("num_threads_training") {
            Some(num_threads_training) => num_threads_training.as_i64().expect("panic since given num_threads_training is not numeric"),
            None => 1
        };

        let params = JsonTypes {
            corpus_file: corpus_file.to_owned(),
            output_dir: output_dir.to_owned(),
            window_size: window_size as i32,
            saved_counts: saved_counts,
            num_threads_cooc: num_threads_cooc as usize,
            json_train: JsonTrain { 
                vocab_size: vocab_size as usize, 
                max_iter: max_iter as usize, 
                embedding_dim: embedding_dim as usize, 
                learning_rate: learning_rate as f32, 
                x_max: x_max as f32, 
                alpha: alpha as f32, 
                batch_size: batch_size as usize, 
                num_threads_training: num_threads_training as usize 
            }
        };

        Ok (
            Self {
                params: params
            }
        )
    }
    
}


pub fn read_input<R: ReadFile>(file_path: &str) -> Result<<R as ReadFile>::Item, <R as ReadFile>::Error> {
    let input = <R as ReadFile>::reaf_file(file_path)?;
    Ok(input)
}

pub fn save_output<S: SaveFile>(output_dir: &str, file_name: &str, item: S) -> Result<(), <S as SaveFile>::Error> {
    
    // create output folder
    if let Err(e) = fs::create_dir_all(output_dir) {
        panic!("{}", e)
    }
    
    // SaveFile can be Array2<f32> or Vec<String>
    item.save_file(output_dir, file_name)?;
    return Ok(())

}

pub trait ReadFile {
    type Error;
    type Item;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error>;
}

impl ReadFile for Vec<u8> {
    type Error = std::io::Error;
    type Item = Self;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
        let in_file = file_path.to_string() + ".json";

        let f = BufReader::new(File::open(in_file)?);
        //let f = File::open(in_file)?;
        let item = serde_json::from_reader(f)?;
        return Ok(item)
    }
}


impl ReadFile for Array2<f32> {
    type Error = ReadNpyError;
    type Item = Self;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {

        let in_file = file_path.to_string() + ".npy";
        let item = read_npy(in_file)?;

        /* 
        let in_file = file_path.to_string() + ".npy.gz";
        let f = BufReader::new(File::open(in_file)?);
        let mut reader = GzDecoder::new(f);
        let mut buf: Vec<u8> = Vec::new();
        reader.read_to_end(&mut buf)?;
        let item: Array2<f32> = bincode::deserialize(&buf).unwrap();
        */
        Ok(item)
    }
}

impl ReadFile for HashMap<String, usize> {
    type Error = std::io::Error;
    type Item = Self;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
        let in_file = file_path.to_string() + ".txt";
        let f = File::open(in_file)?;
        let item = serde_json::from_reader(f)?;
        return Ok(item)
    }
}

impl ReadFile for WrapperCooc {
    type Error = std::io::Error;
    type Item = Self;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
        let in_file = file_path.to_string() + ".txt";
        let f = File::open(in_file)?;
        let item = serde_json::from_reader(f)?;
        return Ok(item)
    }
}

impl ReadFile for Vec<Vec<u8>> {
    type Error = Box<dyn Error>;
    type Item = Self;
    fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
        
        // `read_all_vectored` is not yet available, manually looping throught input for now
        //
        /*
        let in_file = file_path.to_string() + ".gz";
        let f = BufReader::new(File::open(in_file)?);
        let mut reader = GzDecoder::new(f);

        let mut bufs: Vec<IoSliceMut> = Vec::new();
        reader.read_vectored(&mut bufs)?;

        let items: Vec<Vec<u8>> = bufs.iter().map(|buf| {
            buf.to_vec()
        }).collect();
        return Ok(items)
        */
        //
        //

        let mut items: Vec<Vec<u8>> = Vec::new();
        let (main_dir, _) = file_path.rsplit_once("/").unwrap();
        let paths = fs::read_dir(main_dir).unwrap();
        for path in paths {
            let file_path = path.unwrap().path().display().to_string();
            if file_path.ends_with(".gz") {
                let f = BufReader::new(File::open(file_path)?);
                let mut reader = GzDecoder::new(f);
                let mut buf: Vec<u8> = Vec::new();
                reader.read_to_end(&mut buf)?;
                items.push(buf);
            }
        }

        return Ok(items)

    }
}

pub trait SaveFile {
    type Error;
    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error>;
}

impl SaveFile for Vec<Vec<u8>> {
    type Error = Box<dyn Error>;

    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {

        // `write_all_vectored` is not yet available, manually looping throught input for now
        //
        /*
        let out = output_dir.to_string() + "/" + file_name + &format!("{}.gz", i);
        let f = BufWriter::new(File::create(out)?);
        let mut writer = ZlibEncoder::new(f, Compression::default());
        let bufs: Vec<IoSlice> = self.iter().map( |serialized_np_arr| {
            IoSlice::new(serialized_np_arr)
        }).collect();
        writer.write_vectored(&bufs)?;
        Ok(())
        */
        //
        //
        for (i, buf) in self.iter().enumerate() {

            let out = output_dir.to_string() + "/" + file_name + &format!("{}.gz", i);
            let f = BufWriter::new(File::create(out)?);
            let mut writer = GzEncoder::new(f, Compression::default());

            let slice = IoSlice::new(buf);
            writer.write_all(&slice)?;
            writer.flush()?;

        }
        
        Ok(())

    }
}

impl SaveFile for HashMap<(usize, usize), f32> {
    type Error = Box<dyn Error>;

    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {

        let out = output_dir.to_string() + "/" + file_name + ".csv";
        let mut wrt = csv::WriterBuilder::new().from_path(out)?;
        wrt.write_record(&["Token", "Context", "Cooc"])?;

        for ((i, j), v) in self {
            wrt.serialize((i, j, v))?;
        }
        wrt.flush()?;
        Ok(())

    }
}

impl SaveFile for Vec<u8> {
    type Error = Box<dyn Error>;
    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
        let out = output_dir.to_string() + "/" + file_name + ".json";
        let mut f = BufWriter::new(File::create(out)?);
        bincode::serialize_into(&mut f, self)?;
        //serde_json::to_writer(f, self)?;
        return Ok(())
    }
}

impl SaveFile for Array2<f32> {
    type Error = Box<dyn Error>;
    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
        
        let out = output_dir.to_string() + "/" + file_name + ".npy";
        write_npy(out, self)?;

        /* 
        let out = output_dir.to_string() + "/" + file_name + ".npy.gz";
        let f = BufWriter::new(File::create(out)?);
        let mut writer = GzEncoder::new(f, Compression::default());
        let encoded: Vec<u8> = bincode::serialize(&self).unwrap();
        writer.write_all(&encoded)?;
        */
        Ok(())
    }    
}


impl SaveFile for HashMap<String, usize> {
    type Error = std::io::Error;
    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
        let out = output_dir.to_string() + "/" + file_name + ".txt";
        let f = File::create(out)?;
        serde_json::to_writer(f, self)?;
        return Ok(())
    }
}

impl SaveFile for WrapperCooc { // save to txt file to preserve space here ?
    type Error = Box<dyn Error>;
    fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
        
        let out = output_dir.to_string() + "/" + file_name + ".bin";

        let mut f = BufWriter::new(File::create(out)?);
        bincode::serialize_into(&mut f, self)?;

        //let f = OpenOptions::new()
        //.create(true).append(true).open(out)?;

        // check the option to use txt instead...
        //serde_json::to_writer(f, self)?;
        
        return Ok(())
    }
}



#[derive(Clone)]
pub struct WrapperCooc {
    pub x_mat: HashMap<(usize, usize), f32>,
}

impl Serialize for WrapperCooc {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer {

            let cooc_size = self.x_mat.len();
            let mut seq = serializer.serialize_seq(Some(cooc_size))?;
            for (indexes, value) in &self.x_mat {
                let line = vec![indexes.0 as f32, indexes.1 as f32, *value];
                seq.serialize_element(&line)?;
            }
            seq.end()
    }
}

struct HashMapVisitor;
impl<'de> Visitor<'de> for HashMapVisitor {

    type Value = HashMap<(usize, usize), f32>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("vec of size 3 with f32")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>, {

                let mut k2v: HashMap<(usize, usize), f32> = HashMap::new();
                while let Some(line) = seq.next_element::<Vec<f32>>()? {
                    let i = *line.get(0).unwrap() as usize;
                    let j = *line.get(1).unwrap() as usize;
                    let val = *line.get(2).unwrap();
                    k2v.insert((i, j), val);
                }

                Ok(k2v)

    }


}

impl<'de> Deserialize<'de> for WrapperCooc {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de> {
        
            let x_mat = deserializer.deserialize_seq(HashMapVisitor)?;
            Ok( WrapperCooc { x_mat: x_mat } )

    }
}