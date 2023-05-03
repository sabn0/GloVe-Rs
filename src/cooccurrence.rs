
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::ops::Range;
use std::error::Error;
use ndarray::{Array2, Array1, array, s};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::{prelude::*, ThreadPoolBuilder};
use bincode::serialize;
use super::config::{JsonTypes, self};

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

    fn parse_line(line: String, use_os: bool) -> String {
        // remove intial and trailing spaces, move to lower case
        // add end-of-sequence and start-of-sequence tokens
        let mut line_str = [&line.trim().to_lowercase()].map(|x| x.to_string()).to_vec().join(" ");
        if use_os {
            line_str.push_str(" EOS");
            line_str.insert_str(0, "SOS ");
        }
        line_str
    }

    fn accumulate(line: String, token2count: &mut HashMap<String, usize>, sequences: &mut Vec<Vec<String>>, use_os: bool) {

        let sequence = Counts::parse_line(line, use_os);
        let split_sequence = Counts::tokenize(&sequence);

        // accumulate occurences of words
        for tok in &split_sequence {

            let val = token2count.entry(tok.to_owned()).or_insert(0);
            *val += 1;
        }

        sequences.push(split_sequence);

    }

    fn load(file_path: &str, sequences: &mut Vec<Vec<String>>, token2count: &mut HashMap<String, usize>, use_os: bool) -> Result<(), Box<dyn Error>> {

        let lines = Counts::read_file(file_path)?;        
        for line in lines {            
            Counts::accumulate(line?, token2count, sequences, use_os)
        }
        Ok(())
    }


    fn build_vocab(token2count: HashMap<String, usize>, vocab_size: usize, t2i: &mut HashMap<String, usize>, use_shuffle: bool) {
        
        // get most frequenct tokens after sorting
        let mut tup = token2count
        .iter()
        .map(|(k,v)| (k.to_owned(), v))
        .collect::<Vec<(String, &usize)>>();
        tup.sort_by_key(|k| k.1);
        tup.reverse();
        tup.truncate(vocab_size);
        
        if use_shuffle {
            tup.shuffle(&mut thread_rng());
        }
        
        // populate t2i with the most frequent tokens
        (0..vocab_size).into_iter().for_each(|i| { t2i.entry((&tup[i].0).to_owned()).or_insert(i);});

        println!("using {} most common tokens out of {}", vocab_size, token2count.len());

    }


    fn count(window_size: usize, 
        sequences: &Vec<Vec<String>>, 
        tup2cooc: &mut HashMap<(usize, usize), f32>, 
        t2i: &HashMap<String, usize>, 
        slice: &Range<usize>, 
        thread_i: Option<usize>) -> Result<(), Box<dyn Error>> {

        // update x_mat for the co-occurences of a pivot token and context tokens in window
        // the contribution is relative to the distance, 1/d
        let window_size = window_size as i32;

        let n = sequences.len();
        for (_k, sequence) in sequences.iter().enumerate() {

            if thread_i.is_some() && _k % 200000 == 0 {
                println!("within thread {}, counted {}",thread_i.unwrap(), 100.0*(_k as f32 / n as f32));
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
                for j in i+1..=window_size+i {

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


    fn map_to_ndarray(tup2cooc: &HashMap<(usize, usize), f32>, nd_array: &mut Array2<f32>) {

        for (i, (k, v)) in tup2cooc.iter().enumerate() {
            let line: Array1<f32> = array![k.0 as f32, k.1 as f32, *v];
            nd_array.slice_mut(s![i, ..]).assign(&line);
        }
    }


    fn run_thread(window_size: usize, sequences: &Vec<Vec<String>>, t2i: &HashMap<String, usize>, slice: &Range<usize>, thread_i: usize) -> Vec<u8> {

        println!("thread {}, working on vocab slice {:?}", thread_i, slice);
        let mut tup2cooc: HashMap<(usize, usize), f32> = HashMap::new();

        if let Err(e) = Counts::count(window_size, &sequences, &mut tup2cooc, &t2i, slice, Some(thread_i)) {
            panic!("{}", e);
        };

        println!("thread {} founds n rows: {}", thread_i, &tup2cooc.len());

        let mut save_item: Array2<f32> = Array2::from_elem((tup2cooc.len(), 3), 0.0);
        Counts::map_to_ndarray(&tup2cooc, &mut save_item);
        let save_item = serialize(&save_item).expect("could not serialize nd array");

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
        let use_os = true; // wrap sequences with SOS and EOS
        let use_shuffle = true;

        let mut sequences = Vec::new();
        let mut token2count: HashMap<String, usize> = HashMap::new();
        Counts::load(input_file, &mut sequences, &mut token2count, use_os)?;

        // set vocab size to be the min between vocab size and the number of unique tokens found
        if token2count.len() < vocab_size {
            println!("only {} tokens in corpus...", token2count.len());
            vocab_size = token2count.len();
        }

        // now build matrices and vocab
        let mut t2i: HashMap<String, usize> = HashMap::new();
        Counts::build_vocab(token2count, vocab_size, &mut t2i, use_shuffle);

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
        config::files_handling::save_output::<Vec<Vec<u8>>>(&params.output_dir, "cooc", counts_by_slices)?;
        println!("saved as zip files");

        // save the tokens
        config::files_handling::save_output::<HashMap<String, usize>>(&params.output_dir, "words", t2i.clone())?;
        
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


#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use crate::cooccurrence::Counts;

    #[test]
    fn cooc_test() {
        
        // create some dummy data
        let sentences = [
            "What you say makes a lot of sense to me , you are right",
            "Are you playing basketball just for fun or are you a pro ?"
        ];

        let vocab_size = 3; // only top 3 most common, [are, you, a]
        let window_size = 10; // words to the right
        let slice = 0..vocab_size; // do in one pass
        let use_os = false;
        let use_shuffle = false;

        // counts are saved one-sided, symmetry happens before training
        let mut tup2count_golden: HashMap<(usize, usize), f32> = HashMap::new();
        tup2count_golden.insert((0 ,0), 1.0 / 10.0 + 1.0 / 8.0);
        tup2count_golden.insert((0 ,1), 1.0 / 1.0 + 1.0 / 7.0);
        tup2count_golden.insert((0 ,2), 1.0 / 3.0 + 1.0 / 9.0 + 1.0 / 1.0);
        tup2count_golden.insert((1 ,0), 1.0 / 1.0 + 1.0 / 9.0 + 1.0 / 1.0);
        tup2count_golden.insert((1 ,1), 1.0 / 8.0);
        tup2count_golden.insert((1 ,2), 1.0 / 10.0 + 1.0 / 2.0);
        tup2count_golden.insert((2 ,0), 1.0 / 7.0);
        tup2count_golden.insert((2 ,1), 1.0 / 8.0);

        // run algorithm result
        let mut token2count_alg = HashMap::new();
        let mut sequences = Vec::new();
        for line in sentences {
            Counts::accumulate(line.to_string(), &mut token2count_alg, &mut sequences, use_os);
        }
        let mut t2i_alg: HashMap<String, usize> = HashMap::new();
        Counts::build_vocab(token2count_alg, vocab_size, &mut t2i_alg, use_shuffle);

        let mut tup2cooc_alg: HashMap<(usize, usize), f32> = HashMap::new();
        if let Err(e) = Counts::count(window_size, &sequences, &mut tup2cooc_alg, &t2i_alg, &slice, None) {
            panic!("{}", e);
        };

        assert_eq!(tup2cooc_alg, tup2count_golden);

    }

}