//! Minimal dense linear algebra for the contextual learner.
//!
//! Just enough to do Bayesian linear regression in low dimension: symmetric
//! positive-definite inverse, Cholesky factorization (for Gaussian sampling),
//! and matrix-vector / dot products. Matrices are row-major `Vec<Vec<f64>>`;
//! dimensions are small (the feature dimension `d`), so `O(d^3)` is fine.
//!
//! Index-based loops are the natural form here (a single step touches two rows
//! or a triangular range), so the needless-range-loop lint is silenced.
#![allow(clippy::needless_range_loop)]

/// Dot product of two equal-length vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product `m * v`.
pub fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Inverse of a square matrix via Gauss-Jordan elimination with partial
/// pivoting. Panics if the matrix is singular.
pub fn inverse(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    // Augment [m | I].
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        assert!(a[pivot][col].abs() > 1e-12, "matrix is singular");
        a.swap(col, pivot);

        let diag = a[col][col];
        for x in a[col].iter_mut() {
            *x /= diag;
        }
        for r in 0..n {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    for c in 0..2 * n {
                        a[r][c] -= factor * a[col][c];
                    }
                }
            }
        }
    }

    // The right half is the inverse.
    a.iter().map(|row| row[n..].to_vec()).collect()
}

/// Lower-triangular Cholesky factor `l` with `l * l^T == m`, for a symmetric
/// positive-definite `m`. Panics if `m` is not positive-definite.
pub fn cholesky(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                assert!(s > 1e-12, "matrix is not positive-definite");
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();
        let k = b.len();
        let mut out = vec![vec![0.0; m]; n];
        for (i, oi) in out.iter_mut().enumerate() {
            for (j, oij) in oi.iter_mut().enumerate() {
                for l in 0..k {
                    *oij += a[i][l] * b[l][j];
                }
            }
        }
        out
    }

    fn approx_eq(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) -> bool {
        a.iter()
            .zip(b)
            .all(|(ra, rb)| ra.iter().zip(rb).all(|(x, y)| (x - y).abs() < tol))
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let m = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let inv = inverse(&m);
        assert!(approx_eq(&matmul(&m, &inv), &identity(3), 1e-9));
    }

    #[test]
    fn cholesky_reconstructs_the_matrix() {
        let m = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];
        let l = cholesky(&m);
        // l is lower-triangular.
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert_eq!(l[i][j], 0.0);
            }
        }
        // l * l^T == m.
        let lt: Vec<Vec<f64>> = (0..3).map(|i| (0..3).map(|j| l[j][i]).collect()).collect();
        assert!(approx_eq(&matmul(&l, &lt), &m, 1e-9));
    }

    #[test]
    fn matvec_and_dot() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(matvec(&m, &[1.0, 1.0]), vec![3.0, 7.0]);
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }
}
