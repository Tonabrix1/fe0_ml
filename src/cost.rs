use ndarray::{Array};

// enum storing each cost function
pub enum Cost {
    SE,
}

impl Cost {
    pub fn calculate(predicted: Array2<f32>, expected: Array2<f32>) -> Array2<f32>{
        Cost::SE => (predicted - expected).powi(2),
    }

    pub fn derivate(predicted: Array2<f32>, expected: Array2<f32>) -> Array2<f32> {
        2 * (predicted - expected)
    }
}
