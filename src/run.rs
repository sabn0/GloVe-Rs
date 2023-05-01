

use ndarray::Array2;
use super::config;
use super::cooccurence;
use super::train;
use core::panic;
use std::env;
use std::time::Instant;


pub struct Run {}

impl Run {

    pub fn run() {

        println!("entering program...");
        let args: Vec<String> = env::args().collect();
        //let args = vec!["".to_string(), "args_expans.json".to_string()];

        println!("building parameters...");
        let params = match config::Config::new(&args) {
            Ok(config) => config.get_params(), // safe to unwrap now
            Err(e) => panic!("{}", e)
        };

        // run the co-occurences count stage if not saved already
        if params.saved_counts.is_none() || params.saved_counts.unwrap() == false {

            println!("{}", params);
            println!("starting vocab building...");
            let my_time = Instant::now();
    
            if let Err(e) = cooccurence::Counts::run(&params) {
                panic!("{}", e)
            }
            
            println!("finished vocab creation, took {} seconds ...", my_time.elapsed().as_secs());

        }

        // run training part
        let my_time = Instant::now();
        println!("starting training part...");

        // the co-ocurences were saved in parts or given as input, load them
        let cooc_path = (&params.output_dir).to_string() + "/cooc";
        let slices: Vec<Array2<f32>> = match config::read_input::<Vec<Vec<u8>>>(&cooc_path) {
            Ok(slices) => {
                slices.iter().map(|slice| {
                    bincode::deserialize(slice).expect("could not deserialize to nd array")
                }).collect::<Vec<Array2<f32>>>()
            },
            Err(e) => panic!("{}", e)
        };

        println!("loaded {} chunks of cooc", &slices.len());
        let trainer = match train::Train::run(slices, &params) {
            Ok(trainer) => trainer,
            Err(e) => panic!("{}", e)
        };

        let w_tokens = trainer.get_w_tokens();
        let w_context = trainer.get_w_context();
        let w = w_tokens + w_context;
        // this should be the matrix to sample from and compute similarities..

        // save the weights
        if let Err(e) = config::save_output::<Array2<f32>>(&params.output_dir, "vecs", w) { panic!("{}", e) }

        println!("finished training, took {} seconds ...", my_time.elapsed().as_secs());
    
    }

}