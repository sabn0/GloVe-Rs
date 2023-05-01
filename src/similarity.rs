
use std::{error::Error, ops::Range, collections::HashMap};
use ndarray::{prelude::*, concatenate};
use ndarray_stats::*;
use rand::{thread_rng, seq::IteratorRandom};
use plotters::{prelude::*, style::text_anchor::*};

#[allow(dead_code)]
pub struct Similarity {
    w: Array2<f32>,
    t2i: HashMap<String, usize>,
    i2t: HashMap<usize, String>
}

#[allow(dead_code)]
impl Similarity {

    pub fn new(w: &mut Array2<f32>, t2i: HashMap<String, usize>) -> Similarity {
        
        // w is of shape (vocab_size, embedding_dim)

        // need to normalize w so each entry norm l2 is 1
        for mut row in w.axis_iter_mut(Axis(0)) {
            let norm = row.mapv(|a| a.abs().powi(2)).sum().sqrt();
            row.mapv_inplace(|a| (a / norm).abs());
        }
        assert!(*w.max().expect("w is not between 0 and 1 after l2 norm") <= 1.0);
        assert!(*w.min().expect("w is not between 0 and 1 after l2 norm") >= 0.0);

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

    pub fn draw_tokens_2d(&self, save_to: &str, k: usize, use_tokens: Option<Vec<String>>) -> Result<(), Box<dyn Error>> {
      
        // Instead of implementating a PCA I am going to plot on the x axis the mean of the vectors
        // and on the y axis the max of the vectors.

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

        // x axis is removed thus doesn't need much space compared to y axis
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

        // a closure for token label and position
        let position_and_word = |x: f32, y: f32, token: String| {
            return EmptyElement::at((x,y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
                + Text::new(
                    token,
                    (0, 10),
                    &text_style,
                );
        };

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

    pub fn slice_weights(&self, k: usize, indices: Option<Vec<usize>>) -> Result<(Vec<String>, Array2<f32>), Box<dyn Error>> {

        assert_eq!(self.w.dim().0, self.t2i.len(), "inconsistent number of entries in w and tokens");
        assert!(k > 0, "k most be positive");

        let indices = match indices {
            Some(indices) => indices,
            None => (0..self.t2i.len()).choose_multiple(&mut thread_rng(), k)
        };
        
        // slice w and tokens ( maintain order ?)
        let sliced_w: Array2<f32> = self.w.select(Axis(0), &indices);
        let tokens: Vec<String> = indices.iter().map(|i| {
            self.i2t.get(i)
            .expect(format!("did not find token that matches index {}", i).as_str())
            .to_string()
        }).collect();

        assert_eq!(tokens.len(), sliced_w.dim().0);
        return Ok((tokens, sliced_w))

    }

    pub fn get_2dim_projections(&self, w: &Array2<f32>) -> Result<(Array2<f32>, [f32; 4]), Box<dyn Error>> {

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

        // by input order should be 0 = -, 1 = +, 2 = +
        // put (-) of the first item
        let mut sum_analogy: Array1<f32> = -1.0 * self.extract_vec_from_word(inputs[0])?;
        for i in 1..inputs.len() { // add 1 and 2 to sum
            sum_analogy += &self.extract_vec_from_word(inputs[i])?;
        }
        Ok(sum_analogy)
    }

    pub fn extract_analogies(&self, inputs: [&str; 3], k: usize) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
        
        let analogy = self.extract_analogy_vec(inputs)?;
        let best_analogies = self.find_k_most_similar(&analogy, k)?;
        Ok(best_analogies)
    }

    pub fn extract_vec_from_word(&self, token: &str) -> Result<Array1<f32>, Box<dyn Error>> {

        match self.t2i.get(token) {
            Some(i) => return Ok(self.w.slice(s![*i, ..]).to_owned()),
            None => return Err(format!("token: {} is not in most frequent tokens", token).into())
        };

    }

    pub fn find_k_most_similar(&self, vec: &Array1<f32>, k: usize) -> Result<Vec<(String, f32)>, Box<dyn Error>> {

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


#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use std::fs;
    use ndarray::Array2;
    use crate::config;
    use crate::similarity::Similarity;
    
    const WEIGHTS_PATH: &str = "Output/vecs";
    const TOKENS_PATH: &str = "Output/words";

    fn read_weights() -> Array2<f32> {
        match config::files_handling::read_input::<Array2<f32>>(WEIGHTS_PATH) {
            Ok(w) => w,
            Err(e) => panic!("{}", e)
        }
    }

    fn read_t2i() -> HashMap<String, usize> {
        match config::files_handling::read_input::<HashMap<String, usize>>(TOKENS_PATH) {
            Ok(t2i) => t2i,
            Err(e) => panic!("{}", e)
        }
    }

    #[test]
    fn analogies_test() {

        // read weights and tokens
        let mut w = read_weights();
        let t2i = read_t2i();

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        let k = 20; // find the 3 best analogies each time

        // a is to b as like c is to ?
        // translates to b - a + c : ?
        // high is to higher as like good is to : better
        let inputs = [
            ["king", "queen", "man", "woman"],
            ["go", "goes", "say", "says"],
            ["going", "went", "spending", "spent"],
            ["good", "better", "high", "higher"],
            ["child", "children", "dollar", "dollars"],
            ["his", "her", "boy", "girl"],
            ["horse", "horses", "finger", "fingers"],
            ["find", "finds", "eat", "eats"],
            ["employees", "company", "officers", "police"],
            ["after", "before", "big", "small"],
            ["citizens", "president", "children", "parents"],
            ["new", "old", "good", "bad"]
        ];

        for input in inputs {
            
            let source = [input[0], input[1], input[2]];
            let target = input[3];
            let analogies = match sim_obj.extract_analogies(source, k) {
                Ok(analogies) => analogies,
                Err(e) => panic!("{}", e)
            };
            for (i, (analogy, score)) in analogies.iter().enumerate() {
                println!("{} : {} - {} + {} ? {} = {}", i, input[1], input[0], input[2], analogy, score);
            }

            println!("re-computation of the target, might have been found already...");
            let analogy_vec = sim_obj.extract_analogy_vec(source).unwrap(); // safe, it was computed already before
            match sim_obj.extract_vec_from_word(target) {
                Ok(target_vec) => {
                    let target_score = target_vec.dot(&analogy_vec);
                    println!("{} = {}", target, target_score);
                },
                Err(_e) => {
                    println!("supposed target {} is not in the vocabulray", target);
                }
            };


        }


    }

    #[test]
    fn find_most_similar_test() {

        // read weights and tokens
        let mut w = read_weights();
        let t2i = read_t2i();

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        let tokens = ["sun", "student", "basketball", "tree", "singing", "drove", "pretty", "surprised"];
        let k = 10;

        for token in tokens {

            println!("searching {} most similar words to {}", k, token);
            let vec = match sim_obj.extract_vec_from_word(token) {
                Ok(vec) => vec,
                Err(e) => panic!("{}", e)
            };

            match sim_obj.find_k_most_similar(&vec, k) {
                Ok(analogies) => {
                    for (i, (analogy, score)) in analogies.iter().enumerate() {
                        println!("{} : {} ? {} = {}", i, token, analogy, score);  
                    }
                }, 
                Err(e) => panic!("{}", e)
            };
        }
    }

    #[test]
    fn plot_tokens_rand() {

        // read weights and tokens
        let mut w = read_weights();
        let t2i = read_t2i();

        // create output folder
        let output_dir = "Img";
        if let Err(e) = fs::create_dir_all(output_dir) { panic!("{}", e) }

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        let saved_to = format!("{}/2d-rand.png", output_dir);
        if let Err(e) = sim_obj.draw_tokens_2d(&saved_to, 100, None) {
            panic!("{}", e);
        }
    }

    #[test]
    fn plot_tokens_manual() {

        // read weights and tokens
        let mut w = read_weights();
        let t2i = read_t2i();

        // create output folder
        let output_dir = "Img";
        if let Err(e) = fs::create_dir_all(output_dir) { panic!("{}", e) }

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        let saved_to = format!("{}/2d-manual.png", output_dir);
        let tokens = [
            "girl", "boy", "new", "old", "young", "dog", "dogs", "hold", "holds", "woman", "man", "tall", "short"
        ].iter().map(|x| x.to_string()).collect::<Vec<String>>();
        if let Err(e) = sim_obj.draw_tokens_2d(&saved_to, 100, Some(tokens)) {
            panic!("{}", e);
        }
    }
}