

use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_stats::QuantileExt;
use rand::thread_rng;
use crate::config::JsonTrain;
use std::error::Error;
use std::iter::zip;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::slice::Chunks;
use std::time::Instant;


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

impl Train {

    fn new(vocab_size: usize, embedding_dim: usize) -> Train {

        Self {
            w_tokens: Array::random((vocab_size, embedding_dim), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            w_context: Array::random((vocab_size, embedding_dim), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            b_tokens: Array::random((vocab_size, 1), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            b_context: Array::random((vocab_size, 1), Uniform::new(-0.5, 0.5)) / embedding_dim as f32,
            ag_w_tok: Array2::from_elem((vocab_size, embedding_dim), 1.0), // init to 1.0 makes the initial eta equal to inital learning rate
            ag_w_context: Array2::from_elem((vocab_size, embedding_dim), 1.0),
            ag_b_tok: Array2::from_elem((vocab_size, 1), 1.0),
            ag_b_context: Array2::from_elem((vocab_size, 1), 1.0)
        }
    }

    pub fn get_w_tokens(&self) -> Array2<f32> {
        return self.w_tokens.clone();
    }

    pub fn get_w_context(&self) -> Array2<f32> {
        return self.w_context.clone();
    }

    fn weighting_x(xs: &mut Array2<f32>, x_max: f32, alpha: f32) {
        xs.mapv_inplace(|x| { 
            if x < x_max {
                (x / x_max).powf(alpha)
            } else { 
                1.0
            }
        });
    }

    fn do_training_slice(&mut self, 
           slice_arr: &Array2<f32>,
           train_params: &JsonTrain,
           progress_params: &mut DisplayProgress,
        ) -> Result<(), Box<dyn Error>> {

            // train parameters extraction
            let batch_size = train_params.batch_size;
            let x_max = train_params.x_max;
            let alpha = train_params.alpha;
            let learning_rate = train_params.learning_rate;

            // progress_parameters extraction
            let epoch_loss = &mut progress_params.epoch_loss;
            let total_loss = &mut progress_params.total_loss;
            let n_slices = progress_params.n_slices;
            let slice_enumeration = progress_params.slice_enumeration;
            let current_slice = progress_params.current_slice;
            let slice_n_examples = *progress_params.total_examples.get(current_slice).unwrap() as f32;
            println!("in slice {} / {}, number of total examples here: {}", slice_enumeration, n_slices, slice_n_examples);

            let slice_len = slice_arr.dim().0;
            let mut in_slice_order = (0..slice_len).into_iter().collect::<Vec<usize>>();
            in_slice_order.shuffle(&mut thread_rng());

            // split slice to chunks
            let slice_chunks_indexes: Chunks<usize> = in_slice_order.chunks(batch_size);

            for (pp, chunk_indexes) in slice_chunks_indexes.enumerate() {

                // last batch can be smaller than batch_size
                let this_batch = chunk_indexes.len();
                let c_bar = 100000;
                if pp % c_bar == 0 && pp > 0 {
                    let progress = (((this_batch * pp) as f32 / slice_n_examples) * 100.0).floor();
                    println!("in slice {} / {}, {}%, loss: {}", slice_enumeration, n_slices, progress, *epoch_loss / *total_loss);
                }

                // these are the indexes to take from the slice and work on
                let get_indexes = |col: usize| -> Vec<usize> {
                    slice_arr
                    .select(Axis(0), chunk_indexes)
                    .slice(s![.., col])
                    .to_shape(this_batch)
                    .unwrap()
                    .to_vec()
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>()
                };
                let is = &get_indexes(0);
                let js = &get_indexes(1);
                let xs: Array2<f32> = slice_arr.select(Axis(0), chunk_indexes)
                .slice(s![.., 2usize]).map(|x| *x).to_shape((this_batch, 1))?.to_owned();

                // dimensions of (this_batch, embedding_dim) for v
                // dimensions of (this_batch, 1) for b
                // select is not mutable
                let v_tok: Array2<f32> = self.w_tokens.select(Axis(0), is);
                let v_context: Array2<f32> = self.w_context.select(Axis(0), js);
                let b_tok: Array2<f32> = self.b_tokens.select(Axis(0), is);
                let b_context: Array2<f32> = self.b_context.select(Axis(0), js);                

                // get the rows of the ag gradients
                // dimensions of (this_batch, embedding_dim) for v
                // dimensions of (this_batch, 1) for b
                // select is not mutable
                let mut g_v_tok: Array2<f32> = self.ag_w_tok.select(Axis(0), is);
                let mut g_v_context: Array2<f32> = self.ag_w_context.select(Axis(0), js);
                let mut g_b_tok: Array2<f32> = self.ag_b_tok.select(Axis(0), is);
                let mut g_b_context: Array2<f32> = self.ag_b_context.select(Axis(0), js);

                // weight and move to nd array to speed computations, shape (this_batch, 1)
                let mut xs_weighted: Array2<f32> = xs.clone();
                Train::weighting_x(&mut xs_weighted, x_max, alpha);

                // w_tok * w_context should be of shape => (this_batch, emebedding_dim) * (this_batch, emebedding_dim) => (this_batch, 1)
                // diff is of size (this_batch, 1)
                let dp: Array2<f32> = (&v_tok * &v_context).sum_axis(Axis(1)).to_shape((this_batch, 1))?.to_owned();
                let diff: Array2<f32> = &dp + &b_tok + &b_context - &xs.mapv(f32::ln);

                // compute loss for batch
                // need to be careful with sum here, it can reach total_examples.sum() / batch size
                // so basically the batch size needs to be larger than n_slices to avoid a worst case collapse
                let local_batch_loss: Array2<f32> = 0.5 * &xs_weighted * &diff.mapv(|x| x.powi(2));
                let local_loss = local_batch_loss.mean().unwrap();
                *epoch_loss += local_loss;
                *total_loss += 1.0;

                // dl_dw_tok is (this_batch, embedding_dim)
                // dl_db is(this_batch, 1)
                let dl_dw = &xs_weighted * &diff;
                let dl_dw_tok: Array2<f32> = &v_context * &dl_dw;
                let dl_dw_context: Array2<f32> = &v_tok * &dl_dw;
                let dl_db: Array2<f32> = dl_dw;

                // the full derivative updates
                // if sqrt is zero somthing went wrong...
                let dw_tok_update: &Array2<f32> = &(learning_rate * &dl_dw_tok / &g_v_tok.mapv(f32::sqrt));
                let dw_context_update: &Array2<f32> = &(learning_rate * &dl_dw_context / &g_v_context.mapv(f32::sqrt));
                let db_tok_update: &Array2<f32> = &(learning_rate * &dl_db / &g_b_tok.mapv(f32::sqrt));
                let db_context_update: &Array2<f32> = &(learning_rate * &dl_db / &g_b_context.mapv(f32::sqrt));

                // the full grad update
                g_v_tok = &dl_dw_tok * &dl_dw_tok;
                g_v_context = &dl_dw_context * &dl_dw_context;
                g_b_tok = &dl_db * &dl_db;
                g_b_context = &dl_db * &dl_db;

                // update by index,
                // done in a loop since no select_mut by non-consecutive indexes is available
                for (ll, (ii, jj)) in zip(is, js).enumerate() {
                    // weights update
                    self.w_tokens.slice_mut(s![*ii, ..]).sub_assign(&dw_tok_update.slice(s![ll, ..]));
                    self.w_context.slice_mut(s![*jj, ..]).sub_assign(&dw_context_update.slice(s![ll, ..]));
                    self.b_tokens.slice_mut(s![*ii, ..]).sub_assign(&db_tok_update.slice(s![ll, ..]));
                    self.b_context.slice_mut(s![*jj, ..]).sub_assign(&db_context_update.slice(s![ll, ..]));
                    // grad update
                    self.ag_w_tok.slice_mut(s![*ii, ..]).add_assign(&g_v_tok.slice(s![ll, ..]));
                    self.ag_w_context.slice_mut(s![*jj, ..]).add_assign(&g_v_context.slice(s![ll, ..]));
                    self.ag_b_tok.slice_mut(s![*ii, ..]).add_assign(&g_b_tok.slice(s![ll, ..]));
                    self.ag_b_context.slice_mut(s![*jj, ..]).add_assign(&g_b_context.slice(s![ll, ..]));

                }
            }

        Ok(())

    }


    fn train(&mut self, x_mat: Vec<Array2<f32>>, train_params: &JsonTrain) -> Result<(), Box<dyn Error>> {

        let mut progress_params = DisplayProgress::new();
        progress_params.set_n_slices(x_mat.len());
        progress_params.set_total_examples((&x_mat).iter().map(|x| x.dim().0).collect());
        
        for _epoch in 0..train_params.max_iter {

            let my_time = Instant::now();

            // for each epoch - > shuffle the slices order and shuffle the order with each slice
            let mut slices_order = (0..progress_params.n_slices).into_iter().collect::<Vec<usize>>();
            slices_order.shuffle(&mut thread_rng());

            // could be changes to par_iter for threads
            // ThreadPoolBuilder::new().num_threads(params.num_threads_training).build_global()?;
            slices_order.iter().enumerate().for_each( | (rr, m)| {

                progress_params.set_slice_enumeration(rr+1);
                progress_params.set_current_slice(*m);
                let slice_arr = x_mat.get(*m).unwrap(); // shape (N, 3)                

                if let Err(e) = self.do_training_slice(slice_arr, train_params, &mut progress_params) {
                    panic!("{}", e);
                }

            });

            println!("finished epoch {}, loss is {}, took: {} seconds...", _epoch, progress_params.epoch_loss / progress_params.total_loss, my_time.elapsed().as_secs());
        }

        Ok(())

    }

    pub fn run (x_mat_slices: Vec<Array2<f32>>, train_params: &JsonTrain) -> Result<Train, Box<dyn Error>> {
                
        // calculate vocab_size from slices
        let vocab_size = 1 + (&x_mat_slices).iter().map(|x_mat_slice| {
            *x_mat_slice.slice(s![.., ..2usize]).max().unwrap() as usize
        }).max().unwrap();

        // rebuild symmetric part of slices
        // move from the old vector...
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

        let mut trainer = Train::new(vocab_size, train_params.embedding_dim);
        trainer.train(x_mat_symmetric_slices, train_params)?;
        Ok(trainer)
    }


}
