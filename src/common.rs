use ndarray::prelude::*;

#[allow(non_snake_case)]
pub struct System {
    pub n: usize,
    pub x: Array2<f64>,
    pub v: Array2<f64>,
    pub m: Array1<f64>,
    pub G: f64,
}

impl System {
    pub fn center(&mut self) {
        let inv_m_tot = 1. / self.m.sum();
        let x_cm = (&self.x * &self.m.slice(s![.., NewAxis])).sum_axis(Axis(0)) * inv_m_tot;
        let v_cm = (&self.v * &self.m.slice(s![.., NewAxis])).sum_axis(Axis(0)) * inv_m_tot;
        self.x -= &x_cm.slice(s![NewAxis, ..]);
        self.v -= &v_cm.slice(s![NewAxis, ..]);
    }

    pub fn calc_rs(&self) -> (Array3<f64>, Array2<f64>) {
        // Calculate the radius vectors, rv_ij
        let rv_ij = &self.x.slice(s![.., NewAxis, ..]) - &self.x.slice(s![NewAxis, .., ..]);
        // Calculate the scalar radii
        let r_ij = Array2::from_shape_vec(
            (self.n, self.n),
            rv_ij.axis_iter(Axis(0))
                .map(|mat| {
                    mat.axis_iter(Axis(0))
                        .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>()
                .into_iter()
                .flatten()
                .collect(),
        )
        .expect("Dimension mismatch");
        // Return the data
        (rv_ij, r_ij)
    }

    pub fn accel(&mut self, a: &mut Array2<f64>) {
        // Clear the accelerations
        a.fill(0.);
        // Calculate the radii, vectors and scalars
        let (rv_ij, r_ij) = self.calc_rs();
        // Calculate the inverse cubes of the scalar radii: 1./r_ij^3
        let mut inv_r3 = r_ij.mapv(|a| a.powi(-3));
        // Replace the infinities on the diagonal with zeros
        inv_r3.diag_mut().fill(0.);
        // Calculate the accelerations
        a.assign(&(self.G * (&rv_ij * &inv_r3.slice(s![.., .., NewAxis]) * &self.m.slice(s![.., NewAxis, NewAxis])).sum_axis(Axis(0))));
    }

    pub fn energy(&self) -> Array1<f64> {
        let ke = (0.5 * &self.v.mapv(|x| x * x).sum_axis(Axis(1)) * &self.m).sum();
        let m2d = self.m.slice(s![.., NewAxis]);
        let (_rv_ij, r_ij) = self.calc_rs();
        let pe: f64 = (self.G * &r_ij.mapv(|v| 1./v).triu(1) * &m2d * &m2d.t()).sum();
        array![ke, pe]
    }

}
