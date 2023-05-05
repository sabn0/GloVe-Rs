# GloVe Trainer in Rust

![example workflow](https://github.com/Sabn0/GloVe-Rs/actions/workflows/rust.yml/badge.svg)

This is a 100% rust binary code to train a GloVe model based on the details in GloVe's paper [].
Written to experiement, and get some impression of how to implement such an NLP model in rust.
The training part is a three stage pipeline - data preprocessing, coocurrences counting, training.
Testing enables some inspection of word analogies given trained word vectors.

 ## How to run
 After cloning the repo, build and run the main.rs binary in release:
 ```
 cargo build --release
./target/release/main args.json
 ```
 The main.rs binary expects a single argument, a json file. It will specify (at the minimum) two parameters: a txt file with corpus
 of sentences for training, and a location for output directory. 
 ```javascript
 {
 	"corpus_file": "Input/some_sentences.txt",
	"output_dir": "Output",
 }
 ```
By specifying these minimal inputs, the program runs with its default parameters. It will save coocurrence counts to a tar zip,
trained vector embeddings in a vec.npy file, and token-to-int dict to a txt file, all saved in the `output_dir` folder. 
You can modify all or part of the following parameters by adding them to the json:
```javascript
    "window_size": positive integer
    "vocab_size": positive integer
    "max_iter": positive integer
    "embedding_dim": posititve integer
    "batch_size": positive integer
    "x_max": positive integer
    "learning_rate": float between 0-1
    "alpha": float between 0-1
    "num_threads_cooc": positive integer
    "num_threads_training": positive integer
    "saved_counts": boolean
```
`num_threads_cooc` specifies how many thread to use during coocurrence counting. `num_threads_training` is the training 
equivilanet, but the code does not support parallel threads for the training part at the moment (so avoid changing this).
`saved_counts` should be set to true if coocurrence part was already ran and you want only training based on saved files.

### Visualize some relations
The test.rs binary can be used to print some word similarities and analogies based on a trained token embeddings. It expects 4 arguments:
```
    -- selector letter: one char, a/b
    -- input file: txt file with inputs (see examples in the Input directoty)
    -- trained embeddings: npy file (main.rs output)
    -- tokens: txt file with token-to-int dict (main.rs output)
```

"a" is a word analogy test. In the input file, each line should have 4 tokens separated by space. The test will print the 10
closest words (based on cosine distance) to the combination of the first 3 tokens in each line. For example, a line in the input file could be the well-known analogy: king queen man woman. An example to build and run:
 ```
 cargo build
 cargo run --bin test a Input/analogies.txt Output/vecs.npy Output/words.txt
 ```

"b" is a test for word similarites. In the input file, each line should have one single token. The test will print the 10 closest words to each anchor token (baed on cosine distance). An example to build and run:
 ```
 cargo build
 cargo run --bin test b Input/sim_targets.txt Output/vecs.npy Output/words.txt
 ```

## Implementation details
### Preprocessing
Very simple, each line in the corpus file is stripped for leading and trailing spaces, then lower-cased, and wrapped with SOS and
EOS tokens. Tokenization is done by spliting on spaces.
### Coocurrence
First counts the occurrences of all unique tokens, then creates a vocabulary using the requested vocab_size = N most common tokens, and finally counts cooccurrences between the vocabulary elements within the corpus, following GloVe's details. The coccurrences counting part is done using M passes over the corpus, each pass counts the coocurrences between a portion of the vocab and the other words. This serves the porpuse of allowing large vocabulary without memory issues. I set M to 30K, which represents a worst case of 900M entries of token and context pairs. Each portion is saved into an nd array, serialized and compressed. The output is one tar.gz with N / M files.
### Training
First loads the coocurrences from the tar back to M nd arrays, then runs training following GloVe's details. Done in one thread. The training is done in slices that are based on the calculted M arrays. In each epoch, the order of the slices is randomized, and the order within each slice is also randomized. Within each slice, examples are devided to batches based on requested batch_size. When done iterating, the trained weights are saved to a vecs.npy file in the output_dir location.

## Testing
I tested the code using WikiText-103 dataset []. 
After removing headlines and empty lines it has ~100M tokens, which translated to a vocabulary of ~230K tokens when split by spaces. 
Running in release, Coocurrence counting took ~4 minutes using 4 threads. The output tar weighted about 700MB. I then trained for 10 epochs, each epoch took ~18 minutes using a single thread. The output npy weighted about 275MB.
I did not run a full word analogy test after training, but I did inspect some general inputs for sanity. For example I got: 

The 5 most similar words to student :
0 : student ? student = 0.99999994
1 : student ? graduate = 0.8040225
2 : student ? faculty = 0.78390074
3 : student ? students = 0.77575016
4 : student ? undergraduate = 0.72798145
5 : student ? academic = 0.7142711

The 5 most similar words to singing : 
0 : singing ? singing = 0.9999999
1 : singing ? dancing = 0.8588408
2 : singing ? sang = 0.8120471
3 : singing ? sing = 0.80949867
4 : singing ? performing = 0.7759678
5 : singing ? madonna = 0.76943535

king to queen is like man to ? : (excluding king, queen and man as possible answers)
queen - king + man ? woman = 0.77241313

go to goes is like say to says ? : (excluding go, goes and say as possible answers)
goes - go + say ? says = 0.7782096

## Additional notes
I consistently used 32 bit variables in the implementation. Using a numerical gradient check I discovered that for low epsilons
the difference between the x+e , x-e when approximating the gradients vanishes due to 32bit precision. Maybe I will move this to 64
bit in the future, potentially also allowing bigger slices.

## References
This is a rust implementation of the architecture described in the paper GloVe[].
I got some inspiration by nice ideas in this[] python implementation.
I tested after training using WikiText-103[].
I could upload this binary to crates.io if needed.

## License

