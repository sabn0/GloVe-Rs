
use std::{error::Error, ops::Range, collections::HashMap};
use ndarray::{prelude::*};
// use ndarray_linalg::{Eigh, UPLO};
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

        // need to normalize w so each entry norm l2 is 1
        for mut row in w.axis_iter_mut(Axis(0)) {
            let norm = row.mapv(|a| a.abs().powi(2)).sum().sqrt();
            row.mapv_inplace(|a| (a / norm).abs());
        }

        let w_max = *w.max().unwrap();
        let w_min = *w.min().unwrap();
        assert!(w_max <= 1.0);
        assert!(w_min >= 0.0);

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

    pub fn draw_tokens_2d(&self, save_to: &str, k: usize) -> Result<(), Box<dyn Error>> {
      
        const MARGIN: u32 = 15;
        const FONT_STYLE: (&str, i32) = ("sans-serif", 15);

        // get plotting data
        let (tokens, w) = self.select_random_tokens(k)?;
        let projection = self.get_2dim_projections("first_two", &w)?;
        let w_max = *projection.max()?;
        let w_min = *projection.min()?;
        assert!(w_max <= 1.0);
        assert!(w_min >= 0.0);

        let root_area = BitMapBackend::new(save_to, (640, 640)).into_drawing_area();
        root_area.fill(&WHITE)?;

        // weights are normalized thus values between 0 and 1
        let x_spec: Range<f32> = Range{start: w_min, end: w_max};
        let y_spec: Range<f32> = Range{start: w_min, end: w_max};

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
                    (10, 10),
                    &text_style,
                );
        };

        for i in 0..tokens.len() {
            let positions = projection.slice(s![i, ..]).to_slice().unwrap();
            let token = tokens.get(i).unwrap();
            chart.plotting_area().draw(&position_and_word(positions[0], positions[1], token.to_string()))?;
        }

        chart.plotting_area().present()?;
        Ok(())

    }

    pub fn get_2dim_projections(&self, p_type: &str, w: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {

        match p_type {
            "first_two" => {
                let o: Range<usize> = 0..2;
                let sliced = w.slice(s![.., o]);
                return Ok(sliced.to_owned());
            },/*
            "pca" => {

                // centeralizing the weights
                let means = w.mean_axis(Axis(0)).unwrap();
                for mut row in w.axis_iter_mut(Axis(0)) {
                    row -= &means;
                }

                // calculate eigenvalues and eigenvectors
                let cov = w.t().dot(w).mapv(|x| x * (1 / w.dim().0) as f32);
                let (eigs, vecs) = cov.eigh(UPLO::Lower).unwrap();
                
                // get the first and second largest eigs positions
                // could be modified with ------------ndarray_stats --------------
                let mut top_eigs: [f32; 2] = [-f32::INFINITY + f32::EPSILON, -f32::INFINITY];
                let mut vec_indices: [usize; 2] = [0, 0];
                for (i, eig_value) in eigs.iter().enumerate() {
                    if *eig_value > top_eigs[0] {
                        top_eigs[0] = *eig_value;
                        vec_indices[0] = i;
                        top_eigs[1] = top_eigs[0];
                        vec_indices[1] = vec_indices[0];
                    } else if *eig_value > top_eigs[1] {
                        top_eigs[1] = *eig_value;
                        vec_indices[1] = i;
                    }
                }
                let largest_vecs = vecs.select(Axis(1), &vec_indices);
                
                // projection of the vectors to first to pcs
                // this array is of size (k,2) 
                let projection = w.dot(&largest_vecs);
                return Ok(projection)
            },*/
            _ => panic!("unrecognized pattern {}", p_type)
        }


    }

    pub fn select_random_tokens(&self, k: usize) -> Result<(Vec<String>, Array2<f32>), Box<dyn Error>> {

        assert_eq!(self.w.dim().0, self.t2i.len(), "inconsistent number of entries in w and tokens");
        assert!(k>0, "k most be positive");

        // choose random k indices between 0 and vocab size
        let n = self.t2i.len();
        let mut rng = thread_rng();        
        let mut random_indices = (0..n).choose_multiple(&mut rng, k);
        random_indices.sort();

        // slice k and w for those random indices
        let mut w: Array2<f32> = Array2::zeros((k, self.w.dim().1));
        let mut tokens: Vec<String> = Vec::new();
        for (j, i) in random_indices.iter().enumerate() {
            w.slice_mut(s![j, ..]).assign(&self.w.slice(s![*i, ..]));
            let token = self.i2t.get(i).unwrap();
            tokens.push(token.to_owned());
        }

        assert_eq!(tokens.len(), w.dim().0);
        return Ok((tokens, w))

    }

    pub fn extract_analogy_vec(&self, inputs: [&str; 3]) -> Result<Array1<f32>, Box<dyn Error>> {

        let mut vecs: Vec<Array1<f32>> = Vec::new();
        for e in inputs {
            let vec = self.extract_vec_from_word(e)?; 
            vecs.push(vec);
        }

        let analogy = vecs.get(1).unwrap() - vecs.get(0).unwrap() + vecs.get(2).unwrap();
        Ok(analogy)
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

        // multiply all vectors by token vector
        let mut sim_tokens: Vec<(String, f32)> = Vec::new();
        let scores = self.w.dot(vec); // of size w.0, vocab size
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().map(|x| x.to_owned()).enumerate().collect();

        // sort by most similar in descending order
        indexed_scores.sort_by(|(_i, s), (_j, t)| t.total_cmp(s));

        // get k most similar tokens
        for i in 0..k {
            let (index, score) = indexed_scores.get(i).unwrap();
            let sim_tok = self.i2t.get(index).unwrap().to_string(); // safe to unwrap
            sim_tokens.push((sim_tok, *score));
        }

        Ok(sim_tokens)
    }

}


#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use std::fs;

    use crate::similarity::Similarity;
    use ndarray::Array2;
    use crate::config;

    const WEIGHTS_PATH: &str = "Output/vecs";
    const TOKENS_PATH: &str = "Output/words";

    #[test]
    fn analogies_test() {

        // read weights and tokens
        let t2i = config::read_input::<HashMap<String, usize>>(TOKENS_PATH).unwrap();
        let mut w = config::read_input::<Array2<f32>>(WEIGHTS_PATH).unwrap();

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
            let analogies = sim_obj.extract_analogies(source, k).unwrap();
            for (i, (analogy, score)) in analogies.iter().enumerate() {
                println!("{} : {} - {} + {} ? {} = {}", i, input[1], input[0], input[2], analogy, score);    
            }

            let target_vec = sim_obj.extract_vec_from_word(target).unwrap();
            let analogy_vec = sim_obj.extract_analogy_vec(source).unwrap();
            let target_score = target_vec.dot(&analogy_vec);
            println!("{} = {}", target, target_score);

        }


    }

    #[test]
    fn find_most_similar_test() {

        // read weights and tokens
        let t2i = config::read_input::<HashMap<String, usize>>(TOKENS_PATH).unwrap();
        let mut w = config::read_input::<Array2<f32>>(WEIGHTS_PATH).unwrap();

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        let token = "sun";
        let k = 10;
        let vec = sim_obj.extract_vec_from_word(token).unwrap();

        println!("searching {} most similar words to {}", k, token);
        
        match sim_obj.find_k_most_similar(&vec, k) {
            Ok(analogies) => {
                for (i, (analogy, score)) in analogies.iter().enumerate() {
                    println!("{} : {} ? {} = {}", i, token, analogy, score);  
                }
            }, 
            Err(e) => panic!("{}", e)
        };

    }

    #[test]
    fn plot_tokens() {

        // read weights and tokens
        let t2i = config::read_input::<HashMap<String, usize>>(TOKENS_PATH).unwrap();
        let mut w = config::read_input::<Array2<f32>>(WEIGHTS_PATH).unwrap();

        // create output folder
        let output_dir = "Img";
        if let Err(e) = fs::create_dir_all(output_dir) { panic!("{}", e) }

        // run similarity test
        let sim_obj = Similarity::new(&mut w, t2i);
        if let Err(e) = sim_obj.draw_tokens_2d(format!("{}/0.png", output_dir).as_ref(), 100) {
            panic!("{}", e);
        }


    }
}