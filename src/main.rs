mod train;
mod cooccurence;
mod config;
mod similarity;
mod run;

// github link
// comments
// tests
// wiki data
// plotters -> avg value and max value
// check results
// add option for threads during training
// handle cases of "chunks" in data for counting


fn main() {
    run::Run::run();
}

// file preprocess:
// strip, remove empty lines, to lower, remove headlines (only articles), add SOS EOS
