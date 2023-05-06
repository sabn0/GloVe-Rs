

// imports
use crate::config::{files_handling, Config};
use crate::cooccurrence::Counts;
use crate::train::Train;

use core::panic;
use std::env;
use std::time::Instant;
use ndarray::Array2;
use bincode::deserialize;

pub struct Pipeline {}

impl Pipeline {

    // runs the main procedure of 3 steps -
    // -> configuration of arguments
    // -> cooccurrences counting
    // -> training

    pub fn run() {

        println!("entering program...");
        let args: Vec<String> = env::args().collect();
        
        println!("building parameters...");
        let params = match Config::new(&args) {
            Ok(config) => config.get_params(),
            Err(e) => panic!("{}", e)
        };

        // run the cooccurrences count stage if not saved and given already
        if params.saved_counts.is_none() || params.saved_counts.unwrap() == false {

            let timer = Instant::now();
            println!("{}", params);
            println!("starting vocab building...");
    
            if let Err(e) = Counts::run(&params) {
                panic!("{}", e)
            }
            
            println!("finished creation and saved vocab, took {} seconds ...", timer.elapsed().as_secs());

        }

        // run training part
        let timer = Instant::now();
        println!("starting training part...");

        // the cooccurrences were saved in parts or given as input, load them
        let cooc_path = (&params.output_dir).to_string() + "/cooc";
        let slices: Vec<Array2<f32>> = match files_handling::read_input::<Vec<Vec<u8>>>(&cooc_path) {
            Ok(slices) => {
                slices.iter().map(|slice| {
                    deserialize(slice).expect("could not deserialize to nd array")
                }).collect::<Vec<Array2<f32>>>()
            },
            Err(e) => panic!("{}", e)
        };

        // train (and save trained weights) 
        println!("loaded {} chunks of cooccurrences", &slices.len());
        if let Err(e) = Train::run(slices, &params.json_train, &params.output_dir) {
            panic!("{}", e)
        };

        println!("finished training, saved vecs. Took {} seconds ...", timer.elapsed().as_secs());
    
    }

}