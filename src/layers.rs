#![allow(dead_code, unused_variables)]
use ndarray::Array2;
use crate::activations::Activations;

// struct that can be used to accept layers as arguments generally
pub enum Layers {
    Dense{units: usize, activation: Activations, init_func: String},
}

impl Layers {
    pub fn get_units(&self) -> usize {
        match self {
            Layers::Dense{units, activation, init_func} => {
                units.clone()
            },
        }
    }

    pub fn get_init_func(&self) -> String {
        match self {
            Layers::Dense{units, activation, init_func} => init_func.clone(),
        }
    }

    pub fn forward_propagate(&self, input:  &Array2<f32>, weights: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32> {
        match self {
            Layers::Dense{units, activation, init_func} => {
                input.dot(weights) + bias
            }
        }
    }

    pub fn display(&self) -> String{
        match self {
            Layers::Dense{units, activation, init_func} => format!("Dense Layer - {:?} Units - {:?} activation", units, activation),
        }
    }

    pub fn activate(&self, input: &Array2<f32>) -> Array2<f32>{
        match self {
            Layers::Dense{units, activation, init_func} => activation.activate(input),
        }
    }

    pub fn derivate_activation(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layers::Dense{units, activation, init_func} => activation.derivate(input),
        }
    }
}