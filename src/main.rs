mod train;
mod cooccurrence;
mod config;
mod similarity;
mod run;

// comments
// tests
// wiki data lisence
// includes and erase redundant code, pub vs pri
// uninstall what not used, organize toml
// option to result when possible
// existence of keys and values hash

fn main() {
    run::Run::run();
}

// file preprocess:
// strip, remove empty lines, to lower, remove headlines (only articles), add SOS EOS
// counting is done on lines, sequences of texts, that can be as large as the string buffer allows.
// hence the input should be file of sequences and not one huge chunk of text.
// counting is done in slices of size and zipped to save. In training slices that fit variable 32 are 
// loaded once (as many io calls as slices and not a call for every time we process as slice), so RAM
// should allow it. Training is done in one thread.
// results on 100M look good.
// plotting uses max and mean, should be replaced with PCA