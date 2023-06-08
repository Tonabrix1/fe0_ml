use ndarray::Array2;
pub type Dataset = Vec<Sample>;
pub type BatchedDataset = Vec<Dataset>;
pub type ForwardBatch = Vec<Vec<Vec<Array2<f32>>>>;
#[derive(Clone)]
pub struct Sample(pub Array2<f32>, pub Array2<f32>);