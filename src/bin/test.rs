
use core::panic;
use std::{error::Error, env, fs::File, io::{self, BufRead}};
extern crate glove_trainer;
use glove_trainer::Similarity;


// this module has some check on trained vectors, functionallity to get
// the K most similar words to a given word.
// the K most similar words to a combination of words.
// plotting words to 2d (should be changed to PCA later)
// treated as binary executable so it can be ran independantly from main

fn main() {

    // arguments to this executable should be:
    // a letter selector: "a" for analogies, "b" for word similarity
    // path to input based on selector (quads or singles) 
    // path to trained vecs (npy)
    // path to tokens (txt)
    // example: ... a Input/analogies.txt Output/vecs.npy Output words.txt
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 { panic!("input arguments 2, 3 should be path to npy followed by path to txt"); }
    let selector = &args[1];
    if !["a", "b"].contains(&selector.as_str()) { panic!("unrecognized pattern in first argument {}", &args[1]); }

    // read inputs file
    let open_in_file = File::open(&args[2]).expect("could not open input file");
    let lines = io::BufReader::new(open_in_file).lines();

    // read in trained vecs and tokens
    let w = Similarity::read_weights(&args[3]);
    let t2i = Similarity::read_t2i(&args[4]);
    let sim_obj = Similarity::new(w, t2i);

    // match the input selector
    match selector.as_str() {
        "a" => {
            // the case of analogies task, expects a file in which each line holds quartets, separated by a space. For example:
            // king queen man woman

            let inputs = lines
            .into_iter()
            .map(|line| line.expect("could not read line").split(' ')
            .map(|x| x.to_string())
            .collect::<Vec<String>>())
            .collect::<Vec<Vec<String>>>();

            if let Err(e) = run_analogies(&inputs, 10, sim_obj) {
                panic!("{}", e);
            };
        },
        "b" => {
            // the case of word similarity task, expects a file in which each line has one token.

            let inputs = lines
            .into_iter()
            .map(|line| line.expect("could not read line"))
            .collect::<Vec<String>>();

            if let Err(e) = run_similarity(&inputs, 10, sim_obj) {
                panic!("{}", e);
            };

        },
        _ => panic!("unrecognized pattern in first argument {}", &args[1])
    }


}


fn run_analogies(inputs: &Vec<Vec<String>>, k: usize, similarity_object: Similarity) -> Result<(), Box<dyn Error>> {

    // each element in inputs is an iterator over 4 strings, the object is to find the analogy 
    // of the combination of the first 3, in hope that it would match 4.
    // i.e
    // a is to b as like c is to ?
    // translates to b - a + c : ?
    // i.e : high is to higher as like good is to : better

    for input in inputs {

        assert_eq!(input.len(), 4);

        let source = [input[0].as_str(), input[1].as_str(), input[2].as_str()];
        let target = input[3].as_str();
        
        let analogies = similarity_object.extract_analogies(source, k)?;
        let mut found_target = false;
        for (i, (analogy, score)) in analogies.iter().enumerate() {
            println!("{} : {} - {} + {} ? {} = {}", i, source[1], source[0], source[2], analogy, score);
            if analogy == target {
                found_target = true;
                println!("found target '{}' analogy in place {}", target, 1+i);
            }
        }

        if !found_target {
            println!("target '{}' was not found within the first {} analogies", target, k);
        }

        println!("\n");
    }
    Ok(())

}

fn run_similarity(inputs: &[String], k: usize, similarity_object: Similarity) -> Result<(), Box<dyn Error>> {

    // finding the k most similar words to each of the input tokens

    for token in inputs {

        println!("searching {} most similar words to {}", k, token);
        let vec = similarity_object.extract_vec_from_word(token)?;
        let similarities = similarity_object.find_k_most_similar(&vec, k)?;
        for (i, (similar_token, score)) in similarities.iter().enumerate() {
            println!("{} : {} ? {} = {}", i, token, similar_token, score);  
        }
        println!("\n");
    }

    Ok(())


}
