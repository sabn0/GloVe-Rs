
// imports
use crate::config::files_handling;

use std::{error::Error, ops::Range, collections::HashMap};
use ndarray::{prelude::*, concatenate};
use ndarray_stats::*;
use rand::{thread_rng, seq::IteratorRandom};
use plotters::{prelude::*, style::text_anchor::*};



// this module has functionallity on trained vectors:
// the K most similar words to a given word.
// the K most similar words to a combination of words.
// plotting words to 2d (should be changed to PCA later)


pub struct Similarity {
    w: Array2<f32>,
    t2i: HashMap<String, usize>,
    i2t: HashMap<usize, String>
}


impl Similarity {

    pub fn new(w: &mut Array2<f32>, t2i: HashMap<String, usize>) -> Similarity {
        
        // w is of shape (vocab_size, embedding_dim)
        // w entries are normalized to have a l2 norm of 1
        Similarity::normalize(w);

        // get the reverse map of t2i
        let mut i2t: HashMap<usize, String> = HashMap::new();
        for (t, i) in &t2i {
            i2t.entry(*i).or_insert(t.to_owned());
        }

        Self {
            w: w.clone(),
            t2i: t2i,
            i2t: i2t
        }
    }

    fn normalize(w: &mut Array2<f32>) {
        for mut row in w.axis_iter_mut(Axis(0)) {
            let norm = row.mapv(|a| a.abs().powi(2)).sum().sqrt();
            row.mapv_inplace(|a| (a / norm).abs());
        }
        assert!(*w.max().expect("w is not between 0 and 1 after l2 norm") <= 1.0);
        assert!(*w.min().expect("w is not between 0 and 1 after l2 norm") >= 0.0);
    }

    pub fn read_weights(weight_path: &str) -> Array2<f32> {
        match files_handling::read_input::<Array2<f32>>(weight_path) {
            Ok(w) => w,
            Err(e) => panic!("{}", e)
        }
    }

    pub fn read_t2i(tokens_path: &str) -> HashMap<String, usize> {
        match files_handling::read_input::<HashMap<String, usize>>(tokens_path) {
            Ok(t2i) => t2i,
            Err(e) => panic!("{}", e)
        }
    }

    #[allow(dead_code)]
    pub fn draw_tokens_2d(&self, save_to: &str, k: usize, use_tokens: Option<Vec<String>>) -> Result<(), Box<dyn Error>> {
      
        // this method draws tokens on a 2d plane and saves the img to save_to
        // the k tokens can be sampled randomly, or m tokens can be given by the user.

        // Instead of implementating a PCA I am going to plot on the x axis the mean of the vectors
        // and on the y axis the max of the vectors. SHould be changed sometime.

        const MARGIN: u32 = 15;
        const FONT_STYLE: (&str, i32) = ("sans-serif", 20);

        let (tokens, sliced_w) = match use_tokens {
            Some(use_tokens) => {
                let indices = use_tokens.iter().map(|t| self.t2i.get(t)
                .expect(format!("requested token {} not in vocabulary", t).as_str()).to_owned())
                .collect::<Vec<usize>>();
                self.slice_weights(k, Some(indices))?
            },
            None => {
                self.slice_weights(k, None)?
            }
        };

        // get plotting data, projection should be of shape (k, 2)
        let (projections, axes) = self.get_2dim_projections(&sliced_w)?;

        let root_area = BitMapBackend::new(save_to, (640, 640)).into_drawing_area();
        root_area.fill(&WHITE)?;

        // weights are normalized thus values between 0 and 1
        let x_spec: Range<f32> = Range{start: axes[0], end: axes[1]};
        let y_spec: Range<f32> = Range{start: axes[2], end: axes[3]};

        let mut chart = ChartBuilder::on(&root_area)
        .margin(MARGIN)
        .x_label_area_size(10)
        .y_label_area_size(50)
        .build_cartesian_2d(x_spec, y_spec)?;

        chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .disable_y_axis()
        .draw()?;

        let text_style = TextStyle::from(FONT_STYLE)
        .transform(FontTransform::None)
        .font.into_font().style(FontStyle::Bold)
        .with_color(&BLACK)
        .with_anchor::<RGBColor>(Pos::new(HPos::Center, VPos::Center))
        .into_text_style(chart.plotting_area());

        // a closure for token label and position on the 2d plane
        let position_and_word = |x: f32, y: f32, token: String| {
            return EmptyElement::at((x,y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
                + Text::new(
                    token,
                    (0, 10),
                    &text_style,
                );
        };

        // draw every token on the canves
        for i in 0..tokens.len() {
            let token = tokens.get(i).unwrap().to_string(); // safe, tokens is the enumerator
            // positions should be of shape (2,), from (k, 2)
            let positions: Array1<f32> = projections.slice(s![i, ..]).to_owned();
            assert_eq!(positions.dim(), 2);
            chart.plotting_area().draw(&position_and_word(positions[0], positions[1], token))?;
        }

        chart.plotting_area().present()?;
        Ok(())

    }

    fn slice_weights(&self, k: usize, indices: Option<Vec<usize>>) -> Result<(Vec<String>, Array2<f32>), Box<dyn Error>> {

        // get w and tokens sliced on the requested tokens only

        assert_eq!(self.w.dim().0, self.t2i.len(), "inconsistent number of entries in w and tokens");
        assert!(k > 0, "k most be positive");

        let indices = match indices {
            Some(indices) => indices,
            None => (0..self.t2i.len()).choose_multiple(&mut thread_rng(), k)
        };
        
        // slice w and tokens ( this should maintain order )
        let sliced_w: Array2<f32> = self.w.select(Axis(0), &indices);
        let tokens: Vec<String> = indices.iter().map(|i| {
            self.i2t.get(i)
            .expect(format!("did not find token that matches index {}", i).as_str())
            .to_string()
        }).collect();

        assert_eq!(tokens.len(), sliced_w.dim().0);
        return Ok((tokens, sliced_w))

    }

    #[allow(dead_code)]
    fn get_2dim_projections(&self, w: &Array2<f32>) -> Result<(Array2<f32>, [f32; 4]), Box<dyn Error>> {

        // move from w second ax from embedding_dim to 2
        // corrently computed as the mean and max of the values, but should be replaced with PCA... 

        // w is of shape (k, embedding_dim), should return a slice (k, 2) with the max and mean values
        let (k, _) = &w.dim();

        // means and maxs should be of shape (k, 1)
        let means: Array2<f32> = w.mean_axis(Axis(1)).ok_or("problem in mean")?.to_shape((*k, 1usize))?.to_owned();
        let maxs: Array2<f32> = w.map_axis(Axis(1), |v| { 
            *v.iter()
            .max_by(|x, y| x.partial_cmp(y)
            .expect("problem in get_2dim_projections"))
            .expect("problem in get_2dim_projections")
        }).to_shape((*k, 1usize))?.to_owned();

        // get min and max values for axes
        let x_max = *means.max()?;
        let x_min = *means.min()?;
        let y_max = *maxs.max()?;
        let y_min = *maxs.min()?;

        // create slice (k, 2)
        let projections: Array2<f32>  = concatenate![Axis(1), means, maxs];
        let epsi = 0.00;
        Ok((projections, [x_min-epsi, x_max+epsi, y_min-epsi, y_max+epsi]))
    }


    pub fn extract_analogy_vec(&self, inputs: [&str; 3]) -> Result<Array1<f32>, Box<dyn Error>> {

        // given 3 strings of tokens, compute the analogy linear combination of their vectors.

        // by input order should be 0 = -, 1 = +, 2 = +
        // put (-) of the first item
        let mut sum_analogy: Array1<f32> = -1.0 * self.extract_vec_from_word(inputs[0])?;
        for i in 1..inputs.len() { // add 1 and 2 to sum
            sum_analogy += &self.extract_vec_from_word(inputs[i])?;
        }
        Ok(sum_analogy)
    }

    pub fn extract_analogies(&self, inputs: [&str; 3], k: usize) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
        
        // get the k most similar words (and sim - scores) to the linear combination of inputs 
        let analogy = self.extract_analogy_vec(inputs)?;
        let best_analogies = self.find_k_most_similar(&analogy, k)?;
        Ok(best_analogies)
    }

    pub fn extract_vec_from_word(&self, token: &str) -> Result<Array1<f32>, Box<dyn Error>> {
        // given token string extract the vector of that string from w
        match self.t2i.get(token) {
            Some(i) => return Ok(self.w.slice(s![*i, ..]).to_owned()),
            None => return Err(format!("token: {} is not in most frequent tokens", token).into())
        };

    }

    pub fn find_k_most_similar(&self, vec: &Array1<f32>, k: usize) -> Result<Vec<(String, f32)>, Box<dyn Error>> {

        // given a vector of embedding_dim size, get the k most similar tokens to that vector and the scores
        // done by cosine similarity

        assert!(k < self.w.dim().0, "k most be smaller than the vocabulary, but {} given", k);

        // multiply all vectors by token vector
        let mut sim_tokens: Vec<(String, f32)> = Vec::new();
        let scores = self.w.dot(vec); // of size w.0 <=> vocab size
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().map(|x| x.to_owned()).enumerate().collect();

        // sort by most similar in descending order
        indexed_scores.sort_by(|(_i, s), (_j, t)| t.total_cmp(s));

        // get k most similar tokens
        for i in 0..k {
            let (index, score) = indexed_scores.get(i).unwrap(); // safe, indexed_scores of dim k
            let sim_tok = self.i2t.get(index).unwrap().to_string(); // safe, i2t of dim > k
            sim_tokens.push((sim_tok, *score));
        }

        Ok(sim_tokens)
    }

}
