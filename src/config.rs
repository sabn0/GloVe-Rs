
// imports
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};
use serde_json::Value;
use std::{fs::{self, File}, error::Error, collections::HashMap, fmt::Display};
use std::io::prelude::*;
use flate2::{Compression, read::GzDecoder};
use flate2::write::GzEncoder;

// wrapper around parameters for the program (cocc and train)
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

// wrapper around parameters for the training part
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


// validation of input arguments
pub struct Config {
    params: JsonTypes
}

impl Config {

    pub fn get_params(&self) -> JsonTypes {
        return self.params.clone()
    }

    fn read_json(json_path: &String) -> Value {
        let f = fs::File::open(json_path).expect("cannot open json file");
        let json: Value = serde_json::from_reader(f).expect("cannot read json file");
        json
    }

    // the input json has many fields of hyper parameters, this function mainly performs checks 
    // for data types.
    fn validate(json: Value) -> Result<JsonTypes, Box<dyn Error>> {

        let validate_str = |field: &str| {
            json.get(field)
            .expect(format!("{} was not supplied throught json", field).as_str())
            .as_str()
            .expect(format!("cannot cast {} to string", field).as_str())
        };

        let validate_bool = |field: &str, default_val: Option<bool>| {
            match json.get(field) {
                Some(val) => Some(val.as_bool().expect(format!("panic since given {} is not boolean", field).as_str())),
                None => default_val
            }
        };

        let validate_positive_integer = |field: &str, default_val: u64| {
            let val = match json.get(field) {
                Some(val) => {
                    let val = val.as_u64().expect(format!("panic since given {} = {} is not positive integer", field, val).as_str());
                    assert!(val > 0, "panic since given {} = {} is not positive integer", field, val);
                    val
                },
                None => default_val
            };
            val
        };

        let validate_float = |field: &str, default_val: f64| {
            let val = match json.get(field) {
                Some(val) => {
                    let val = val.as_f64().expect(format!("panic since given {} = {} is not float", field, val).as_str());
                    assert!(val > 0.0, "{}", format!("panic since given {} = {} is not positive float", field, val));
                    assert!(val <= 1.0, "{}", format!("panic since given {} = {} is bigger than 1.0", field, val));
                    val
                },
                None => default_val
            };
            val
        };

        // validate input and output in json - most be given
        let corpus_file = validate_str("corpus_file");
        let output_dir = validate_str("output_dir");

        // handle default vs input parameters
        // requested parameters overwrite the default parameters
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

        Ok(params)

    }

    // program should receive one arguments, path to json
    pub fn new(args: &[String]) -> Result<Config, Box<dyn Error>> { 

        if args.len() != 2 {
            return Err(format!("input should be a path to json file only").into());
        }

        let json = Config::read_json(&args[1]);
        let params = Config::validate(json)?;

        Ok (
            Self {
                params: params
            }
        )
    }
    
}

// this module orgenizes read and write options using different data structures in the program
// currently supporting 
// HashMap<String, usize>   (tokens),
// Array2<f32>              (vecs)
// Vec<Vec<u8>>             (coocs),
pub mod files_handling {

    use std::io::{BufWriter, IoSlice};

    use tar::{Builder, Header, Archive};
    use super::*;

    // this method is called in order to read an input of type R from file_path for supported types
    pub fn read_input<R: ReadFile>(file_path: &str) -> Result<<R as ReadFile>::Item, <R as ReadFile>::Error> {
        let input = <R as ReadFile>::reaf_file(file_path)?;
        Ok(input)
    }
    
    // this method is called in order to write an input of type R for supported types, into output_dir + file_name + ext
    pub fn save_output<S: SaveFile>(output_dir: &str, file_name: &str, item: S) -> Result<(), <S as SaveFile>::Error> {
        
        // create output folder
        if let Err(e) = fs::create_dir_all(output_dir) {
            panic!("{}", e)
        }

        item.save_file(output_dir, file_name)?;
        return Ok(())
    
    }

    // this trait defines the reading behavior types should obey to    
    pub trait ReadFile {
        type Error;
        type Item;
        fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error>;
    }
    
    // this trait defines the writing behavior types should obey to
    pub trait SaveFile {
        type Error;
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error>;
    }
    
    // tokens are read with this implementation
    impl ReadFile for HashMap<String, usize> {
        type Error = std::io::Error;
        type Item = Self;
        fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
            let f = File::open(file_path)?;
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
    // trained vecs are read with this implementation
    impl ReadFile for Array2<f32> {
        type Error = Box<dyn Error>;
        type Item = Self;
        fn reaf_file(file_path: &str) -> Result<Self::Item, Self::Error> {
            let item = read_npy(file_path)?;
            Ok(item)
        }
    }
    // trained vecs are saved with this implementation
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

            let tar_path = format!("{}.tar.gz", file_path);
            let tar_gz = File::open(tar_path)?;
            let tar = GzDecoder::new(tar_gz);
            let mut archive = Archive::new(tar);

            let mut items: Vec<Vec<u8>> = Vec::new();
            for gz_file in archive.entries()? {
                let mut gz_file = gz_file?;
                let mut buf: Vec<u8> = Vec::new();
                gz_file.read_to_end(&mut buf)?;
                items.push(buf);
            }
    
            return Ok(items)
    
        }
    }
    // cooc counts are zipped with this implementation
    impl SaveFile for Vec<Vec<u8>> {
        type Error = Box<dyn Error>;
    
        fn save_file(&self, output_dir: &str, file_name: &str) -> Result<(), Self::Error> {

            let out_path = format!("{}/{}", output_dir, file_name);
            let tar_gz = File::create(format!("{}.tar.gz", &out_path))?;
            let enc = GzEncoder::new(tar_gz, Compression::default());
            let mut tar_builder = Builder::new(enc);

            for (i, buf) in self.iter().enumerate() {

                let mut header = Header::new_gnu();
                header.set_path(format!("{}{}.gz", file_name, i))?;
                header.set_size(buf.len() as u64);
                header.set_cksum();
                                
                tar_builder.append(&mut header, buf.as_slice())?;
            }

            tar_builder.finish()?;  
            Ok(())
    
        }
    }
}


#[cfg(test)]
mod tests {

    // tests mainly validate data types of inputs compliance

    use serde_json::{Value, json};
    use super::Config;

    #[test]
    fn config_test_dedault() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some other string",
        });


        let params = match Config::validate(json) {
            Ok(params) => params,
            Err(e) => panic!("{}", e)
        };

        assert_eq!(params.corpus_file, "some string");
        assert_eq!(params.output_dir, "some other string");
    }

    #[test]
    fn config_test_input() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "window_size": 1,
            "saved_counts": false,
            "num_threads_cooc": 1,
            "vocab_size": 1, 
            "max_iter": 2, 
            "embedding_dim": 3, 
            "learning_rate": 0.21, 
            "x_max": 1, 
            "alpha": 0.01,
            "batch_size": 1, 
            "num_threads_training": 1, 

        });

        let params = match Config::validate(json) {
            Ok(params) => params,
            Err(e) => panic!("{}", e)
        };

        assert_eq!(params.window_size, 1);
        assert_eq!(params.saved_counts, Some(false));
        assert_eq!(params.num_threads_cooc, 1);
        assert_eq!(params.json_train.vocab_size, 1);
        assert_eq!(params.json_train.max_iter, 2);
        assert_eq!(params.json_train.embedding_dim, 3);
        assert_eq!(params.json_train.learning_rate, 0.21);
        assert_eq!(params.json_train.x_max, 1 as f32);
        assert_eq!(params.json_train.alpha, 0.01);
        assert_eq!(params.json_train.batch_size, 1);
        assert_eq!(params.json_train.num_threads_training, 1);



    }


    #[test]
    #[should_panic(expected = "panic since given window_size = 1.0 is not positive integer")]
    fn config_test_window_size() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "window_size": 1.0,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    #[should_panic(expected = "panic since given saved_counts is not boolean")]
    fn config_test_saved_counts() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "saved_counts": 0,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }


    #[test]
    #[should_panic(expected = "panic since given num_threads_cooc = -1 is not positive integer")]
    fn config_test_num_threads_cooc() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "num_threads_cooc": -1,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    #[should_panic(expected = "panic since given vocab_size = 0 is not positive integer")]
    fn config_test_vocab_size() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "vocab_size": 0,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    #[should_panic(expected = "panic since given max_iter = -10.0 is not positive integer")]
    fn config_test_max_iter() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "max_iter": -10.0,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    #[should_panic(expected = "panic since given embedding_dim = 100.4 is not positive integer")]
    fn config_test_embedding_dim() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "embedding_dim": 100.4,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }


    #[test]
    #[should_panic(expected = "panic since given x_max = 1.0 is not positive integer")]
    fn config_test_x_max() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "x_max": 1.0,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    fn config_test_batch_size() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "batch_size": 89,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

    #[test]
    fn config_test_num_threads_training() {

        let json: Value = json!({
            "corpus_file": "some string",
            "output_dir": "some string",
            "num_threads_training": 9,
        });

        if let Err(e)= Config::validate(json) {
            panic!("{}", e);
        }
    }

}