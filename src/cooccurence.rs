
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::ops::Range;
use ndarray::{Array2, Array1, array, s};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use rayon::{prelude::*, ThreadPoolBuilder};
use crate::config::{JsonTypes, self};


pub struct Counts {}

impl Counts {

    fn read_file(file_path: &str) -> Result<Lines<BufReader<File>>, Box<dyn Error>> {

        match File::open(file_path) {
            Ok(f) => {
                let x = io::BufReader::new(f).lines();
                return Ok(x)
            }
            Err(e) => return Err(Box::new(e))
        };
    }

    fn parse_line(line: String) -> String {
        // remove intial and trailing spaces, move to lower case
        // add end-of-sequence and start-of-sequence tokens
        let line_str = ["SOS", &line.trim().to_lowercase(), "EOS"].map(|x| x.to_string()).to_vec().join(" ");
        line_str
    }

    fn load(file_path: &str, sequences: &mut Vec<Vec<String>>) -> Result<HashMap<String, usize>, Box<dyn Error>> {

        let mut token2count: HashMap<String, usize> = HashMap::new();
        let lines = Counts::read_file(file_path)?;
        
        for line in lines {            

            let sequence = Counts::parse_line(line?);
            let split_sequence = Counts::tokenize(&sequence);

            // accumulate occurences of words
            for tok in &split_sequence {

                let val = token2count.entry(tok.to_owned()).or_insert(0);
                *val += 1;
            }

            sequences.push(split_sequence);

        }

        Ok(token2count)

    }


    fn build_vocab(token2count: HashMap<String, usize>, vocab_size: usize, t2i: &mut HashMap<String, usize>) {
        
        //
        let mut known_words = 0;

        // get most frequenct tokens after sorting
        let keys: Vec<&String> = token2count.keys().collect();
        let values: Vec<&usize> = token2count.values().collect();
        let mut tup: Vec<(String, &usize)> = std::iter::zip(keys, values)
        .map(|(a, b)| (a.to_owned(), b))
        .collect();
        
        tup.sort_by_key(|k| k.1);
        tup.reverse();
        tup.truncate(vocab_size);
        tup.shuffle(&mut thread_rng());

        for i in 0..vocab_size {

            let tok = &tup[i].0;
            t2i.entry(tok.to_owned()).or_insert(known_words);
            known_words += 1;
        }

        println!("using {} most common tokens out of {}", vocab_size, token2count.len());

    }


    fn count(window_size: i32, sequences: &Vec<Vec<String>>, tup2cooc: &mut HashMap<(usize, usize), f32>, t2i: &HashMap<String, usize>, slice: &Range<usize>, thread_i: usize) -> Result<(), Box<dyn Error>> {

        if window_size <= 0 {
            return Err(format!("window size {} is not valid", window_size).into());
        }

        // update x_mat for the co-occurences of a pivot token and context tokens in window
        // the contribution is relative to the distance, 1/d

        let n = sequences.len();
        for (_k, sequence) in sequences.iter().enumerate() {

            if _k % 200000 == 0 {
                println!("within thread {}, counted {}", thread_i, 100.0*(_k as f32 / n as f32));
            }

            let n = sequence.len() as i32;

            for i in 0..n {
                
                let tok = &sequence[i as usize];
                let token_i = match t2i.get(tok) {
                    Some(token_i) if slice.contains(token_i)  => *token_i,
                    Some(_token_i) => continue,
                    None => continue
                };

                // build in symmetry
                for j in i+1..window_size+i {

                    if j >=n { break }

                    let context = &sequence[j as usize];
                    let context_j = match t2i.get(context) {
                        Some(context_j) => *context_j,
                        None => continue
                    };


                    let distance = (j-i) as f32;
                    let dis_count = 1.0 / distance;

                    let val = tup2cooc.entry((token_i, context_j)).or_insert(0.0);
                    *val += dis_count;


                }

            }
        }

        Ok(())

    }


    pub fn map_to_ndarray(tup2cooc: &HashMap<(usize, usize), f32>, nd_array: &mut Array2<f32>) {

        for (i, (k, v)) in tup2cooc.iter().enumerate() {
            let line: Array1<f32> = array![k.0 as f32, k.1 as f32, *v];
            nd_array.slice_mut(s![i, ..]).assign(&line);
        }
    }


    pub fn run_thread(window_size: i32, sequences: &Vec<Vec<String>>, t2i: &HashMap<String, usize>, slice: &Range<usize>, thread_i: usize) -> Vec<u8> {

        println!("thread {}, working on vocab slice {:?}", thread_i, slice);
        let mut tup2cooc: HashMap<(usize, usize), f32> = HashMap::new();

        if let Err(e) = Counts::count(window_size, &sequences, &mut tup2cooc, &t2i, slice, thread_i) {
            panic!("{}", e);
        };

        println!("thread {} founds n rows: {}", thread_i, &tup2cooc.len());

        let mut save_item: Array2<f32> = Array2::from_elem((tup2cooc.len(), 3), 0.0);
        Counts::map_to_ndarray(&tup2cooc, &mut save_item);
        let save_item = bincode::serialize(&save_item).expect("could not serialize nd array");

        println!("finished thread {}, vocab slice {:?}", thread_i, slice);
        save_item

    }    

    pub fn run(params: &JsonTypes) -> Result<(), Box<dyn Error>> {

        // max 1,000,000,000 * 3 Array2<f32> sized allowed
        // should slice 30K tokens in vocab slice for the worst case.

        // extract params, safe to unwrap
        let input_file = params.corpus_file.as_ref();
        let mut vocab_size = params.json_train.vocab_size;
        let window_size = params.window_size;

        let mut sequences = Vec::new();
        let token2count = Counts::load(input_file, &mut sequences)?;

        // set vocab size to be the min between vocab size and the number of unique tokens found
        if token2count.len() < vocab_size {
            println!("only {} tokens in corpus...", token2count.len());
            vocab_size = token2count.len();
        }

        // now build matrices and vocab
        let mut t2i: HashMap<String, usize> = HashMap::new();
        Counts::build_vocab(token2count, vocab_size, &mut t2i);

        // count co-occurences
        // counting is done in parts to enable large vocabulary without allocation failure
        // this should be done in threads...
        let in_parts_size: usize = 30000;
        let total: usize = t2i.len();
        let slices: Vec<Range<usize>> = (0..total)
        .step_by(in_parts_size)
        .map(|i| i..i+in_parts_size)
        .collect();
        ThreadPoolBuilder::new().num_threads(params.num_threads_cooc).build_global()?;
        println!("total number of tokens considered: {}", total);

        // collecting the slices counts in serialized format into one vector
        let counts_by_slices: Vec<Vec<u8>> = slices.par_iter().enumerate().map( |(thread_i, slice)| {
            Counts::run_thread(window_size, &sequences, &t2i, slice, thread_i)
        }).collect();

        // save the counts to one zip, should be cheaper
        config::save_output::<Vec<Vec<u8>>>(&params.output_dir, "cooc", counts_by_slices)?;
        println!("saved as zip files");

        // save the tokens
        config::save_output::<HashMap<String, usize>>(&params.output_dir, "words", t2i.clone())?;
        
        Ok(())

    }


}


trait Tokeniizer {
    fn tokenize(sequence: &str) -> Vec<String>;
}

impl Tokeniizer for Counts {
    
    // simple tokenizer by split
    fn tokenize(sequence: &str) -> Vec<String> {
        return sequence.split(' ').map(|x| x.to_string()).collect();
    }
}