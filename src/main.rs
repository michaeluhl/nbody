use std::error::Error;

use ndarray::prelude::*;
use ndarray_npy::write_npy;
use lazy_static::lazy_static;

mod common;
mod ephem;


lazy_static! {
    static ref CONVERSION_FACTOR: f64 = 86400.0_f64.powf(2.) / 149597870.7_f64.powf(3.);
}


fn main() -> Result<(), Box<dyn Error>> {

    static TOTAL_TIME: f64 = 200.0 * 365.24;
    static DT: f64 = 1.0;
    static O_DT: f64 = 0.1 * 365.24;

    let total_increments: usize = (TOTAL_TIME / DT).floor() as usize;
    let o_intervals: usize = (O_DT / DT).floor() as usize;

    let n_steps: usize = total_increments / o_intervals + 1;
    println!("n_steps: {}", n_steps);

    let mut s = common::System {
        n: 10,
        x: arr2(&ephem::POSN),
        v: arr2(&ephem::VEL),
        m: arr1(&ephem::MASS),
        //m: Array1::from_vec(ephem::MASS.to_vec()),
        G: 132712440041.279419 * *CONVERSION_FACTOR,
    };
    s.center();

    println!("G: {}", s.G);

    let mut sol_x: Array3<f64> = Array::zeros((n_steps, s.n, 3));
    let mut sol_t: Array1<f64> = Array::zeros(n_steps);
    let mut sol_e: Array2<f64> = Array::zeros((n_steps, 2));

    let mut a: Array2<f64> = Array::zeros((s.n, 3));

    for i in 0..total_increments {
        if i % o_intervals == 0 {
            // println!("Output at {}, {}, {}, {}", i / o_intervals, n_steps, i, total_increments);
            sol_e.slice_mut(s![i / o_intervals, ..]).assign(&s.energy());
            sol_t[i / o_intervals] = DT * (i as f64);
            sol_x.slice_mut(s![i / o_intervals, .., ..]).assign(&s.x);
        }
        s.accel(&mut a);
        s.v += &(&a * DT);
        s.x += &(&s.v * DT);
    }

    write_npy("sys_x.npy", &sol_x)?;
    write_npy("energy.npy", &sol_e)?;
    Ok(())

}
