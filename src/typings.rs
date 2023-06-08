use ndarray::Array2;
pub type dataset = Vec<Sample>;
pub type ds_batch = Vec<dataset>;
pub type forward_batch = Vec<Vec<Vec<Array2<f32>>>>;
#[derive(Clone)]
pub struct Sample(pub Array2<f32>, pub Array2<f32>);