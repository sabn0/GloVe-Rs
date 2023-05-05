# GloVe Trainer in Rust

![example workflow](https://github.com/Sabn0/GloVe-Rs/actions/workflows/rust.yml/badge.svg)

This is a 100% rust binary code to train a **GloVe** model based on the details in [GloVe's paper](https://aclanthology.org/D14-1162/). I implemented it to experiement with rust in the context of such an NLP model (and it's been great!). The training part is a 3 stage pipeline - (1) data preprocessing, (2) coocurrences counting, (3) training. In addition, I implemented some testing functionalities that enable inspection of word analogies and similarities with the model's output (trained word vectors).

 ## How to run
 After cloning the repo, build and run the *main.rs* binary in release:
 ```
 cargo build --release
./target/release/main args_example.json
 ```
 The *main.rs* binary expects a single argument, a json file. The json should specify (at the minimum) two parameters: (1) a txt file with corpus of sentences for training. (2) a location for output directory. As follows:
 ```javascript
 {
    "corpus_file": "Input/some_sentences.txt",
    "output_dir": "Output"
 }
 ```
By specifying these 2 arguments, the program will run with its default parameters on the `corpus_file`, and save the outputs to `output_dir`. The outputs are: (1) coocurrence counts in a tar.gz archive. (2) trained vector embeddings in a vec.npy file. (3) token-to-int dict in a txt file. You can modify the following parameters by adding them to the json:
```javascript
{
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
}
```
`num_threads_cooc` specifies how many threads will be used during coocurrence counting. `num_threads_training` is the training 
equivilanet, but the code does not support parallel threads for the training part at the moment (so avoid changing this field).
`saved_counts` should be set to true if the coocurrence part was already ran, and you only want to perform training based on a saved tar.gz of coocurrences.

#### Visualize some relations
The *test.rs* binary can be used to print some word similarities and analogies based on trained vectors. It expects 4 arguments:
```
- selector              one char, a/b
- input file            txt file with inputs
- trained vecs          npy file (main.rs output)
- tokens                txt file with token-to-int dict (main.rs output)
```

For a word analogy examination, select a. In the input file, each line should have 4 tokens separated by space (see examples in the Input directoty). The test will print the 10 closest words, based on cosine distance, to the combination of the first 3 tokens in each line. For example, a line in the input file could be the well-known analogy: *king queen man woman*. An example of how to build and run:
 ```
 cargo build
 cargo run --bin test a Input/analogies.txt Output/vecs.npy Output/words.txt
 ```

For a word similarity examination, select b. In the input file, each line should have one single token (see examples in the Input directoty). The test will print the 10 closest words to each token in the file, based on cosine distance. An example of how to build and run:
 ```
 cargo build
 cargo run --bin test b Input/sim_targets.txt Output/vecs.npy Output/words.txt
 ```

## Implementation details
#### Preprocessing
Very simple: Each line in the corpus file is stripped of leading and trailing spaces, then lower-cased, and wrapped with SOS and
EOS tokens. Tokenization is done by spliting on spaces.
#### Coocurrence
First counts the occurrences of all unique tokens. Then, creates a vocabulary using the requested vocab_size = N most common tokens. Finally, counts cooccurrences between the vocabulary elements in the corpus, following GloVe's details. Coccurrences counting is done using M passes over the corpus, each pass counts the coocurrences between a portion of the vocab and the other words. This serves the porpuse of allowing larger vocabularies without memory issues. I set M to 30K, which represents a worst case of 900M entries of token and context pairs. Each portion is saved into an nd array, serialized and compressed. The output is a single tar.gz that contains  N / M files.
#### Training
First loads the coocurrences from the tar back to M nd arrays, then runs training following GloVe's details. Done in one thread. The training is done in slices that are based on the calculted M arrays. In each epoch, the order of the slices is randomized, and the order within each slice is also randomized. Within each slice, examples are devided to batches based on requested batch_size. When done iterating, the trained weights are saved to a vecs.npy file in the output_dir location.

## Testing
I tested the code using the [**WikiText-103 dataset**](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/). After removing headlines and empty lines, I accounted for **~100M tokens**, which translated to a vocabulary of **~230K tokens** after split by space. Here are some performance details based on my experiemnt, running the entire training pipeline in release:

| part | time | N threads | output weight |
| :--: |  :-------: | :-------: | :-------: |
| **coocurrence** | ~ 4 minutes | 4 | tar.gz around 700MB |
| **training**    | ~ 18 minutes per epoch |  1  |  npy around 275MB |

I ran training for 10 epochs. I did not run a full word analogy test after training, but I did inspect some manuall inputs for sanity. Here are some example I got:

<table>
<tr>
<th> The 5 most similar words to student </th>
<th> The 5 most similar words to singing </th>
</tr>
<tr>
<td>

| 0 | student | student | 0.99999994 |
| :--: |  :-------: | :-------: | :-------: |
| 1 | student | graduate | 0.8040225 |
| 2 | student | faculty | 0.78390074 |
| 3 | student | students | 0.77575016 |
| 4 | student | undergraduate | 0.72798145 |
| 5 | student | academic | 0.7142711 |

</td>
<td>

| 0 | singing | singing | 0.9999999 |
| :--: |  :-------: | :-------: | :-------: |
| 1 | singing | dancing | 0.8588408 |
| 2 | singing | sang | 0.8120471 |
| 3 | singing | sing | 0.80949867 |
| 4 | singing | performing | 0.7759678 |
| 5 | singing | madonna | 0.76943535 |

</td>
</tr>

<tr>
<th> king to queen is like man to ? </th>
<th> go to goes is like say to says ? </th>
</tr>
<tr>
<td>

| 0 | queen - king + man = woman | 0.77241313 |
| :--: |  :-------: | :-------: |
| 1 | queen - king + man = girl | 0.6918511 |
| 2 | queen - king + man = mother | 0.6108579 |

(excluding king, queen and man as possible answers)

</td>
<td>

| 0 | goes - go + say = says | 0.7782096 |
| :--: |  :-------: | :-------: |
| 1 | goes - go + say = knows | 0.71007967 |
| 2 | goes - go + say = everything | 0.70047474 |


(excluding go, goes and say as possible answers)

</td>
</tr>
</table>
<br />

## Additional notes
* I consistently used 32 bit variables in the implementation. Using a numerical gradient check I discovered that for low epsilons,
the difference between x+e and x-e approximating the gradients vanished due to the 32bit precision. Maybe I will move to 64
bit in the future, potentially also allowing bigger slices.
* If it would be of any need, I could upload this binary to crates.io.

## References
* This is a rust implementation of the architecture described in the paper [GloVe](https://aclanthology.org/D14-1162/), by <ins>Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014</ins>, [pdf](https://nlp.stanford.edu/pubs/glove.pdf).
* I tested my implementation using the [WikiText-103](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset, by <ins>Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016</ins>, [(paper)](https://arxiv.org/abs/1609.07843?ref=blog.salesforceairesearch.com).
* I assisted with some nice details in the next posts that used python [1](http://www.foldl.me/2014/glove-python/), [2](https://towardsdatascience.com/a-comprehensive-python-implementation-of-glove-c94257c2813d).


## Software
I used rust version 1.67.1, see Cargo.toml file for the used packages.

## License

