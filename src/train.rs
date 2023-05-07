
// imports
use crate::config::{self, JsonTrain};

use ndarray::{prelude::*, concatenate, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_stats::QuantileExt;
use rand::thread_rng;
use std::error::Error;
use std::time::Instant;

// The struct will hold the learned weights and the adagrad updating weights
// Each entry corresponds to a different token in the vocabulary, columns are embedding_dim
pub struct Train {
    w_tokens: Array2<f32>,
    w_context: Array2<f32>,
    b_tokens: Array2<f32>,
    b_context: Array2<f32>,
    ag_w_tok: Array2<f32>,
    ag_w_context: Array2<f32>,
    ag_b_tok: Array2<f32>,
    ag_b_context: Array2<f32>,
}

impl Train {

    fn new(vocab_size: usize, embedding_dim: usize) -> Train {

        // adagrad weights are initalized to 1 so that the first update is only the learning_rate
        Self {
            w_tokens: Array::random((vocab_size, embedding_dim), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            w_context: Array::random((vocab_size, embedding_dim), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            b_tokens: Array::random((vocab_size, 1), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            b_context: Array::random((vocab_size, 1), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            ag_w_tok: Array2::from_elem((vocab_size, embedding_dim), 1.0), // init to 1.0 makes the initial eta = inital learning rate
            ag_w_context: Array2::from_elem((vocab_size, embedding_dim), 1.0),
            ag_b_tok: Array2::from_elem((vocab_size, 1), 1.0),
            ag_b_context: Array2::from_elem((vocab_size, 1), 1.0)
        }
    }

    fn get_w_tokens(&self) -> Array2<f32> {
        return self.w_tokens.clone();
    }

    fn get_w_context(&self) -> Array2<f32> {
        return self.w_context.clone();
    }

    fn compute_loss_and_grads(
        xs: f32,
        x_max: f32,
        alpha: f32,
        v_tok: &ArrayViewMut1<f32>,
        v_context: &ArrayViewMut1<f32>,
        b_tok: f32,
        b_context: f32
     ) -> Result<(f32, (Array1<f32>, Array1<f32>, f32, f32)), Box<dyn Error>> {

        // this part performs both "forward" and "backward" passing for a single (batched) example xs
        // return a tuple of score of forward (loss) for the batch, and gradients for the batch

        // -- start of forward computation --
        // xs is of shape (this_batch, 1)
        let xs_weighted = if xs < x_max { (xs / x_max).powf(alpha) } else { 1.0 };

        // w_tok * w_context should be of shape => (this_batch, emebedding_dim) * (this_batch, emebedding_dim) => (this_batch, 1),
        // doing it by element-wise multiplication followed by sum over the rows. In turn diff is also (this_batch, 1)
        let diff =  v_tok.dot(v_context) + b_tok + b_context - xs.ln();

        // compute loss for batch based on learning formula of GloVe
        let local_batch_loss = 0.5 * xs_weighted * diff.powi(2);
        // -- end of forward computation --

        // -- backward computation --
        // dl_dw_tok, dl_dw_context are (this_batch, embedding_dim)
        // dl_db is (this_batch, 1)
        let dl_dw = xs_weighted * diff;   // let dl_dw = dl_dw.min(100.0).max(-100.0); // need clip?
        let dl_dw_tok = dl_dw * v_context;
        let dl_dw_context = dl_dw * v_tok;
        let dl_db = dl_dw;
        // corresponding to:
        // cost = 0.5 * f(x_i_j) * (w_i.dot(w_j) + b_i + b_j - ln(x_i_j)) **2 )
        // dx_dw_i = f(x_i_j) * (w_i.dot(w_j) + b_i + b_j - ln(x_i_j) * w_j
        // dx_dw_j = f(x_i_j) * (w_i.dot(w_j) + b_i + b_j - ln(x_i_j) * w_i
        // dx_db_i = dx_db_j = f(x_i_j) * (w_i.dot(w_j) + b_i + b_j - ln(x_i_j)

        Ok((local_batch_loss, (dl_dw_tok, dl_dw_context, dl_db, dl_db)))


    }

    fn do_training_slice(&mut self, 
        slice_arr: &Array2<f32>,
        train_params: &JsonTrain,
        progress_params: &mut DisplayProgress,
     ) -> Result<(), Box<dyn Error>> {

         // this method runs a training step to a slice of coocurences that was saved
         // extract the example, run throught forward and graident computation, updates
         // weights and adagrad weights.
         // If later moved to training in threads with locks, this would be done in different threads.

         // train parameters extraction
         let x_max = train_params.x_max;
         let alpha = train_params.alpha;
         let learning_rate = train_params.learning_rate;
         let verbose = train_params.progress_verbose;

         // progress_parameters extraction
         let epoch_loss = &mut progress_params.epoch_loss;
         let total_loss = &mut progress_params.total_loss;
         let n_slices = progress_params.n_slices;
         let slice_enumeration = progress_params.slice_enumeration;
         let current_slice = progress_params.current_slice;
         let slice_n_examples = *progress_params.total_examples.get(current_slice).unwrap() as f32;
         println!("in slice {} / {}, number of total examples here: {}", slice_enumeration, n_slices, slice_n_examples);

         // shuffle the inner order of examples within the slice
         let slice_len = slice_arr.dim().0;
         let mut in_slice_order = (0..slice_len).into_iter().collect::<Vec<usize>>();
         in_slice_order.shuffle(&mut thread_rng());

         // run training step to each batch
         for (pp, example_index) in in_slice_order.iter().enumerate() {

             // print some progress , last batch can be smaller than batch_size
             if verbose && pp % 1000000 == 0 && pp > 0 {
                 let progress = (((1 * pp) as f32 / slice_n_examples) * 100.0).floor();
                 println!("in slice {} / {}, {}%", slice_enumeration, n_slices, progress);
             }

             let example = slice_arr.slice(s![*example_index, ..]); // (3,)
             let is = example[0] as usize;
             let js = example[1] as usize;
             let xs = example[2];
             
             // dimensions of (embedding_dim,) for v_tok, v_context
             // dimensions of (1,) for b_tok, b_context
             let mut v_tok: ArrayViewMut1<f32> = self.w_tokens.slice_mut(s![is, ..]);
             let mut v_context: ArrayViewMut1<f32> = self.w_context.slice_mut(s![js, ..]);
             let b_tok = self.b_tokens.get_mut((is, 0)).ok_or("did not find index")?;
             let b_context = self.b_context.get_mut((js, 0)).ok_or("did not find index")?;             

             let (local_batch_loss, (dl_dw_tok, dl_dw_context, dl_db_tok, dl_db_context)): (f32, (Array1<f32>, Array1<f32>, f32, f32)) = Train::compute_loss_and_grads(xs, x_max, alpha, &v_tok, &v_context, *b_tok, *b_context)?;

             // I am using a mean over the batch since if I would sum over all examples, epoch_loss & total_loss
             // would (in the worst case) have to hold a size that can be above 32bit. By using batch_size avg,
             // this number is bounded to total_examples / batch_size. However, i am not enforcing batch_size > some k
             *epoch_loss += local_batch_loss;
             *total_loss += 1.0;


             // get the rows of the adagrad gradients
             // dimensions of (this_batch, embedding_dim) for v_tok, v_context
             // dimensions of (this_batch, 1) for b_tok, b_context
             let mut g_v_tok: ArrayViewMut1<f32> = self.ag_w_tok.slice_mut(s![is, ..]);
             let mut g_v_context: ArrayViewMut1<f32> = self.ag_w_context.slice_mut(s![js, ..]);
             let g_b_tok = self.ag_b_tok.get_mut((is, 0)).ok_or("did not find index")?;
             let g_b_context = self.ag_b_context.get_mut((js, 0)).ok_or("did not find index")?;    


             // the full derivative updates
             // if sqrt is zero somthing went wrong...
             let dw_tok_update: &Array1<f32> = &(learning_rate * &dl_dw_tok / &g_v_tok.mapv(f32::sqrt));
             let dw_context_update: &Array1<f32> = &(learning_rate * &dl_dw_context / &g_v_context.mapv(f32::sqrt));
             let db_tok_update: f32 = learning_rate * dl_db_tok / (g_b_tok.sqrt());
             let db_context_update: f32 = learning_rate * dl_db_context / (g_b_context.sqrt());

             // the full adagrad update
             g_v_tok += &(&dl_dw_tok * &dl_dw_tok);
             g_v_context += &(&dl_dw_context * &dl_dw_context);
             *g_b_tok += dl_db_tok * dl_db_tok;
             *g_b_context += dl_db_context * dl_db_context;

             // weights update
             v_tok -= dw_tok_update;
             v_context -= dw_context_update;
             *b_tok -= db_tok_update;
             *b_context -= db_context_update;
         }

     Ok(())

 }


    fn train(&mut self, x_mat: Vec<Array2<f32>>, train_params: &JsonTrain) -> Result<(), Box<dyn Error>> {

        // this method runs the training process over the entrie coocs corpus, given in x_mat by slices.
        // It runs the slices in turnes for N epochs. Currently not allowing multiple threads (since locks are needed).
        // slices are shuffled in order, and also within themselves (random inner order).

        let mut progress_params = DisplayProgress::new();
        progress_params.set_n_slices(x_mat.len());
        progress_params.set_total_examples((&x_mat).iter().map(|x| x.dim().0).collect());
        
        for _epoch in 0..train_params.max_iter {

            let my_time = Instant::now();
            progress_params.init_epoch_loss();
            progress_params.init_total_loss();

            // for each epoch -> shuffle the slices order
            let mut slices_order = (0..progress_params.n_slices).into_iter().collect::<Vec<usize>>();
            slices_order.shuffle(&mut thread_rng());

            // If moving to multi-thread later, this line could be easily changed to par_iter with rayon
            // uncomment the following to multi-thread
            // ThreadPoolBuilder::new().num_threads(params.num_threads_training).build_global()?;
            slices_order.iter().enumerate().for_each( | (rr, m)| {

                progress_params.set_slice_enumeration(rr+1);
                progress_params.set_current_slice(*m);
                let slice_arr = x_mat.get(*m)
                .expect(format!("did not find slice {} in vec", *m).as_str()); // shape (N, 3)                

                if let Err(e) = self.do_training_slice(slice_arr, train_params, &mut progress_params) {
                    panic!("{}", e);
                }

            });

            println!("finished epoch {}, loss is {}, took: {} seconds...", _epoch, progress_params.epoch_loss / progress_params.total_loss, my_time.elapsed().as_secs());
        }

        Ok(())

    }

    pub fn run (x_mat_slices: Vec<Array2<f32>>, train_params: &JsonTrain, output_dir: &str) -> Result<(), Box<dyn Error>> {
        
        // this method runs the training procedure - rearranging slices from non-symmetrical to symmetrical, training,
        // and saving the trained vectors

        // calculate vocab_size from slices
        let vocab_size = 1 + (&x_mat_slices).iter().map(|x_mat_slice| {
            *x_mat_slice.slice(s![.., ..2usize]).max().expect("did not find max value in slice") as usize
        }).max().ok_or("did not find max value in all slices")?;

        // rebuild symmetric part of slices
        let mut x_mat_symmetric_slices: Vec<Array2<f32>> = Vec::new();
        for x_mat_slice in x_mat_slices{
            
            let (rows, _) = x_mat_slice.dim();

            let is: Array1<f32> = x_mat_slice.slice(s![.., 0usize]).to_owned();
            let js: Array1<f32> = x_mat_slice.slice(s![.., 1usize]).to_owned();
            let xs: Array1<f32> = x_mat_slice.slice(s![.., 2usize]).to_owned();

            let new_is: Array2<f32> = concatenate![Axis(0), is, js].to_shape((rows*2, 1))?.to_owned();
            let new_js: Array2<f32> = concatenate![Axis(0), js, is].to_shape((rows*2, 1))?.to_owned();
            let new_xs: Array2<f32> = concatenate![Axis(0), xs, xs].to_shape((rows*2, 1))?.to_owned();
            let new_mat_symmetric_slice: Array2<f32>  = concatenate![Axis(1), new_is, new_js, new_xs];

            assert_eq!(&new_mat_symmetric_slice.dim().0, &(x_mat_slice.dim().0 * 2));
            assert_eq!(&new_mat_symmetric_slice.dim().1, &x_mat_slice.dim().1);

            x_mat_symmetric_slices.push(new_mat_symmetric_slice);

        }

        // run training of slices of examples
        let mut trainer = Train::new(vocab_size, train_params.embedding_dim);
        trainer.train(x_mat_symmetric_slices, train_params)?;

        // w the matrix to sample from and compute similarities, taken as the sum of tokens and context vectors
        let w: Array2<f32> = trainer.get_w_tokens() + trainer.get_w_context();

        // save the weights
        config::files_handling::save_output::<Array2<f32>>(output_dir, "vecs", w)?; 

        Ok(())

    }


}

// this implementation mehrly allows convinient printing during training
struct DisplayProgress {
    epoch_loss: f32, // counting the avg loss of a batch
    total_loss: f32, // counting the number of batchs
    n_slices: usize,        // the number of slices x_mat is divided to
    slice_enumeration: usize, // enumerator over n_slices
    current_slice: usize,   // the value (index) of the current slice
    total_examples: Vec<usize>,  // number of training examples per slice
}

impl DisplayProgress {

    fn new() -> Self {
        Self {
            epoch_loss: 0.0,
            total_loss: 0.0,
            n_slices: 0,
            slice_enumeration: 0,
            current_slice: 0,
            total_examples: Vec::new()
        }
    }

    fn init_epoch_loss(&mut self) {
        self.epoch_loss = 0.0;
    }
    fn init_total_loss(&mut self) {
        self.total_loss = 0.0;
    }

    fn set_n_slices(&mut self, n_slices: usize) {
        self.n_slices = n_slices
    }
    fn set_slice_enumeration(&mut self, slice_enumeration: usize) {
        self.slice_enumeration = slice_enumeration
    }
    fn set_current_slice(&mut self, current_slice: usize) {
        self.current_slice = current_slice
    }
    fn set_total_examples(&mut self, total_examples: Vec<usize>) {
        self.total_examples = total_examples
    }

}


#[cfg(test)]
mod tests {

    use ndarray::{Array, array, Array1};
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use ndarray_stats::QuantileExt;
    use super::Train;

    // I include a gradient check test, based on cost and grads formula
    // with the approximation of numerical gradients. 

    #[test]
    fn gradients_test() {

        // test is ran for each x separatly, not in batches (the computation is less heavy that way)
        let x_values = vec![50.0, 0.05, 43.01, 101.1, 0.002];
        for x_val in x_values {

            let input: f32 = x_val;
            let embedding_dim = 100;
            let x_max = 100.0;
            let alpha = 0.75;
            let epsilon = 0.01;
            let vec_dim = [embedding_dim, embedding_dim, 1, 1];

            // weights are initalized as they would in Train::new(), but for a single entry
            let mut w_i = Array::random(vec_dim[0], Uniform::new(-0.5, 0.5)) / embedding_dim as f32;
            let mut w_j = Array::random(vec_dim[1], Uniform::new(-0.5, 0.5)) / embedding_dim as f32;
            let b_i = Array::random(vec_dim[2], Uniform::new(-0.5, 0.5)) / embedding_dim as f32;
            let b_j = Array::random(vec_dim[3], Uniform::new(-0.5, 0.5)) / embedding_dim as f32;
            
             // dimensions of (embedding_dim,) for v_tok, v_context
             // dimensions of (1,) for b_tok, b_context           
             let (_, gradients): (f32, [Array1<f32>; 4]) = match Train::compute_loss_and_grads(input, x_max, alpha, &w_i.view_mut(), &w_j.view_mut(), b_i[0], b_j[0]) {
                Ok(res) => {
                    // pack gradients - a workaround for the loop below
                    let (local_batch_loss, (dw_i, dw_j, db_i, db_j)) = res;
                    let grad_array = [dw_i, dw_j, Array1::from_elem(1, db_i), Array1::from_elem(1, db_j)];
                    (local_batch_loss, grad_array)
                },
                Err(e) => panic!("{}", e)
             };


            let weights: [Array1<f32>; 4] =  [w_i, w_j, b_i, b_j];
            let mut weights_copy = weights.clone();
            // compute aprroximation of gradients based on forward of w+epsilon and forward of w-epsilon
            for i in 0..=3 {

                for r in 0..vec_dim[i] {

                    {
                    let val = weights_copy[i].get_mut(r).ok_or("could not find index").unwrap();
                    *val += epsilon;
                    }
                    let [ref mut copy_w_i, ref mut copy_w_j, ref copy_b_i, ref copy_b_j] = weights_copy;
                    let (plus, _): (f32, (Array1<f32>, Array1<f32>, f32, f32)) = match Train::compute_loss_and_grads(input, x_max, alpha, &copy_w_i.view_mut(), &copy_w_j.view_mut(), copy_b_i[0], copy_b_j[0]) {
                        Ok(res) => res,
                        Err(e) => panic!("{}", e)
                    };

                    {
                    let val = weights_copy[i].get_mut(r).ok_or("could not find index").unwrap();
                    *val -= 2.0*epsilon;
                    }
                    let [ref mut copy_w_i, ref mut copy_w_j, ref copy_b_i, ref copy_b_j] = weights_copy;
                    let (minus, _): (f32, (Array1<f32>, Array1<f32>, f32, f32)) = match Train::compute_loss_and_grads(input, x_max, alpha,&copy_w_i.view_mut(), &copy_w_j.view_mut(), copy_b_i[0], copy_b_j[0]) {
                        Ok(res) => res,
                        Err(e) => panic!("{}", e)
                    };

                    {
                    let val = weights_copy[i].get_mut(r).ok_or("could not find index").unwrap();
                    *val += epsilon;
                    }

                    // compare per parameter
                    let prediction_grad = (plus - minus) / (2.0 * epsilon);
                    let gold_grad = gradients[i][r];

                    let difference = (prediction_grad - gold_grad).abs() / array![1.0, prediction_grad.abs(), gold_grad.abs()].max().unwrap();
                    assert!(difference < epsilon, "failed, difference is {}", difference);
                }
            }
        }


    }
}