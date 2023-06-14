#![allow(dead_code)]
use ndarray::Array2;

use crate::{cost::Cost::MSE, matrixutil::create_weight, activations::Activations::{ReLU, LeakyReLU, Softmax}};
pub fn mse_unit_tests() {
    let m = MSE;
    let mut x = create_weight(&vec![1,10]);
    let mut y = create_weight(&vec![1,10]);
    y[[0,6]] = 1.;
    let z = vec![0.1, 0.31, 0., 0.004, 0.3, 0., 0.41, 0.2, 0.01, 0.];
    for i in 0..10 {
        x[[0, i]] = z[i];
    }
    let outp = m.calculate(&x,&y);
    println!("MSE({}, {}) = {}", x, y, outp);
    assert!((0.05843*10000f32).round() == (outp * 10000f32).round(), "{} does not match {} within 5 decimal point precision", outp, 0.05843);
    // 0.058431607 ~= 0.05843160000000001 (validate.py)
    let outp2 = m.derivate(&x, &y);
    println!("derivative MSE({}, {}) = {}", x, y, outp2);
    assert!((0.0668*1000f32).round() == (outp2 * 1000f32).round(), "{} does not match {} within 4 decimal point precision", outp2, 0.0668);
    // 0.0668 == 0.0668 (validate.py)
    println!("MSE unit test: PASS");
}

pub fn relu_unit_tests() {
    let r = ReLU;
    let mut x = create_weight(&vec![1,10]);
    let mut y: Array2<f32> = create_weight(&vec![1,10]);
    let mut y2: Array2<f32> = create_weight(&vec![1,10]);
    let z = vec![0.1, -0.31, 0., 0.004, 0.3, 0., -0.41, 0.2, 0.01, 0.];
    let z2: Vec<f32> = vec![0.1, 0., 0., 0.004, 0.3, 0., 0., 0.2, 0.01, 0.];
    let z3: Vec<f32> = vec![1., 0., 0., 1., 1., 0., 0., 1., 1., 0.];
    for i in 0..10 {
        x[[0, i]] = z[i];
        y[[0,i]] = z2[i];
        y2[[0, i]] = z3[i];
    }
    let outp = r.activate(&x);
    println!("ReLU({}) = {}", x, outp);
    assert!(outp == y, "{} does not match {}", outp, y);
    let outp2 = r.derivate(&x);
    println!("derivative ReLU({}) = {}", x, outp2);
    assert!(outp2 == y2,"{} doesn't match {}", outp2, y2);
    println!("ReLU unit tests: PASSED");
}

pub fn leakyrelu_unit_tests() {
    let a = 0.02;
    let r = LeakyReLU{ a };
    let mut x = create_weight(&vec![1,10]);
    let mut y: Array2<f32> = create_weight(&vec![1,10]);
    let mut y2: Array2<f32> = create_weight(&vec![1,10]);
    let z = vec![0.1, -0.31, 0., 0.004, 0.3, 0., -0.41, 0.2, 0.01, 0.];
    let z2: Vec<f32> = vec![0.1, -0.31*a, 0., 0.004, 0.3, 0., -0.41*a, 0.2, 0.01, 0.];
    let z3: Vec<f32> = vec![1., a, a, 1., 1., a, a, 1., 1., a];
    for i in 0..10 {
        x[[0, i]] = z[i];
        y[[0,i]] = z2[i];
        y2[[0, i]] = z3[i];
    }
    let outp = r.activate(&x);
    println!("LeakyReLU({}) = {}", x, outp);
    assert!(outp == y, "{} does not match {}", outp, y);
    let outp2 = r.derivate(&x);
    println!("derivative LeakyReLU({}) = {}", x, outp2);
    assert!(outp2 == y2,"{} doesn't match {}", outp2, y2);
    println!("Leaky ReLU unit tests: PASSED");
}

pub fn softmax_unit_tests() {
    let s = Softmax;
    let mut x = create_weight(&vec![1,10]);
    let mut y: Array2<f32> = create_weight(&vec![1,10]);
    let mut y2: Array2<f32> = create_weight(&vec![1,10]);
    let z = vec![0.1, -0.31, 0., 0.004, 0.3, 0., -0.41, 0.2, 0.01, 0.];
    let z2: Vec<f32> = vec![0.1, -0.31*a, 0., 0.004, 0.3, 0., -0.41*a, 0.2, 0.01, 0.];
    let z3: Vec<f32> = vec![1., a, a, 1., 1., a, a, 1., 1., a];
    for i in 0..10 {
        x[[0, i]] = z[i];
        y[[0,i]] = z2[i];
        y2[[0, i]] = z3[i];
    }
    let outp = s.activate(&x);
    println!("Softmax({}) = {}", x, outp);
    assert!(outp == y, "{} does not match {}", outp, y);
    let outp2 = r.derivate(&x);
    println!("derivative Softmax({}) = {}", x, outp2);
    assert!(outp2 == y2,"{} doesn't match {}", outp2, y2);
    println!("Softmax unit tests: PASSED");
}