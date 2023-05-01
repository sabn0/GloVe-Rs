
// imports
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};
use serde_json::Value;
use std::{fs::{self, File}, error::Error, collections::HashMap, fmt::Display};
use std::io::{prelude::*, {BufWriter, BufReader, IoSlice}};
use flate2::{Compression, read::GzDecoder};
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
        write!(f, 
            "training hyper parameters:
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
    pub window_size: usize,
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
        {}",
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

        let validate_str = |field: &str| {
            json.get(field)
            .expect(format!("{} was not supplied throught json", field).as_str())
            .as_str()
            .expect(format!("cannot cast {} to string", field).as_str())
        };

        let validate_bool = |field: &str, default_val: Option<bool>| {
            match json.get(field) {
                Some(field) => Some(field.as_bool().expect(format!("panic since given {} is not boolean", field).as_str())),
                None => default_val
            }
        };

        let validate_positive_integer = |field: &str, default_val: u64| {
            let val = match json.get(field) {
                Some(field) => {
                    let val = field.as_u64().expect(format!("panic since given {} is not integer", field).as_str());
                    assert!(val > 0, "{}", format!("panic since given {} is not positive integer", field));
                    val
                },
                None => default_val
            };
            val
        };

        let validate_float = |field: &str, default_val: f64| {
            let val = match json.get(field) {
                Some(field) => {
                    let val = field.as_f64().expect(format!("panic since given {} is not float", field).as_str());
                    assert!(val > 0.0, "{}", format!("panic since given {} is not positive float", field));
                    assert!(val <= 1.0, "{}", format!("panic since given {} is bigger than 1.0", field));
                    val
                },
                None => default_val
            };
            val
        };

        // validate input and output in json
        let corpus_file = validate_str("input_file");
        let output_dir = validate_str("output_dir");

        // handle default vs input parameters
        let vocab_size = validate_positive_integer("vocab_size", 400000);
        let max_iter = validate_positive_integer("max_iter", 50);
        let embedding_dim = validate_positive_integer("embedding_dim", 300);
        let x_max = validate_positive_integer("x_max", 100);
        let batch_size = validate_positive_integer("batch_size", 32);
        let num_threads_cooc = validate_positive_integer("num_threads_cooc", 4);
        let num_threads_training = validate_positive_integer("num_threads_training", 1);
        let window_size = validate_positive_integer("window_size", 10);
        let learning_rate = validate_float("learning_rate", 0.05);
        let alpha = validate_float("alpha", 0.75);
        let saved_counts = validate_bool("saved_counts", None);

        let params = JsonTypes {
            corpus_file: corpus_file.to_owned(),
            output_dir: output_dir.to_owned(),
            window_size: window_size as usize,
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

pub mod files_handling {

    use super::*;

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
    
    pub trait SaveFile {
        type Error;
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error>;
    }
    
    // tokens are read with this implementation
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
    // tokens are saved with this implementation
    impl SaveFile for HashMap<String, usize> {
        type Error = std::io::Error;
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
            let out = output_dir.to_string() + "/" + file_name + ".txt";
            let f = File::create(out)?;
            serde_json::to_writer(f, self)?;
            return Ok(())
        }
    }
    // trained vec are read with this implementation
    impl ReadFile for Array2<f32> {
        type Error = Box<dyn Error>;
        type Item = Self;
        fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
            let in_file = file_path.to_string() + ".npy";
            let item = read_npy(in_file)?;
            Ok(item)
        }
    }
    // trained vec are saved with this implementation
    impl SaveFile for Array2<f32> {
        type Error = Box<dyn Error>;
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
            let out = output_dir.to_string() + "/" + file_name + ".npy";
            write_npy(out, self)?;
            Ok(())
        }    
    }
    // cooc counts are unzipped with this implementation
    impl ReadFile for Vec<Vec<u8>> {
        type Error = Box<dyn Error>;
        type Item = Self;
        fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
            
            //
            // `read_all_vectored` is not yet available, manually looping throught input for now
            //  this is the future implementation :
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
            let (main_dir, _) = file_path.rsplit_once("/").ok_or("invalid path file")?;
            let paths = fs::read_dir(main_dir)?;
            for path in paths {
                let file_path = path?.path().display().to_string();
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
    // cooc counts are zipped with this implementation
    impl SaveFile for Vec<Vec<u8>> {
        type Error = Box<dyn Error>;
    
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {
            //
            // `write_all_vectored` is not yet available, manually looping throught input for now
            // this is the future implementation :
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
}
