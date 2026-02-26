//////////////////////////////////////////////////////////////////////////////////////
// *** Courtesy to https://github.com/neka-nat/fastmunk for main implementation *** //
//////////////////////////////////////////////////////////////////////////////////////

use fixedbitset::FixedBitSet;
use ndarray::ArrayView2;
use numpy::{PyArray2, PyArrayMethods};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

fn kuhn_munkres(weights: ArrayView2<f64>, maximize: bool) -> Vec<(usize, usize)> {
    // We call x the rows and y the columns. (nx, ny) is the size of the matrix.
    let nx = weights.nrows();
    let ny = weights.ncols();

    assert!(
        nx <= ny,
        "number of rows must not be larger than number of columns"
    );
    let copied_weights = if !maximize {
        weights.map(|x| -x).view().to_owned()
    } else {
        weights.to_owned()
    };
    // xy represents matching for x, yz matching for y
    let mut xy: Vec<Option<usize>> = vec![None; nx];
    let mut yx: Vec<Option<usize>> = vec![None; ny];
    // lx is the labelling for x nodes, ly the labelling for y nodes. We start
    // with an acceptable labelling with the maximum possible values for lx
    // and 0 for ly.
    let mut lx: Vec<OrderedFloat<f64>> = (0..nx)
        .map(|row| {
            (0..ny)
                .map(|col| OrderedFloat(copied_weights[(row, col)]))
                .max()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut ly: Vec<f64> = vec![0.0; ny];
    // s, augmenting, and slack will be reset every time they are reused. augmenting
    // contains Some(prev) when the corresponding node belongs to the augmenting path.
    let mut s = FixedBitSet::with_capacity(nx);
    let mut alternating = Vec::with_capacity(ny);
    let mut slack = vec![0.0; ny];
    let mut slackx = Vec::with_capacity(ny);
    for root in 0..nx {
        alternating.clear();
        alternating.resize(ny, None);
        // Find y such that the path is augmented. This will be set when breaking for the
        // loop below. Above the loop is some code to initialize the search.
        let mut y = {
            s.clear();
            s.insert(root);
            // Slack for a vertex y is, initially, the margin between the
            // sum of the labels of root and y, and the weight between root and y.
            // As we add x nodes to the alternating path, we update the slack to
            // represent the smallest margin between one of the x nodes and y.
            for y in 0..ny {
                slack[y] = lx[root].0 + ly[y] - copied_weights[(root, y)];
            }
            slackx.clear();
            slackx.resize(ny, root);
            Some(loop {
                let mut delta = f64::INFINITY;
                let mut x = 0;
                let mut y = 0;
                // Select one of the smallest slack delta and its edge (x, y)
                // for y not in the alternating path already.
                for yy in 0..ny {
                    if alternating[yy].is_none() && slack[yy] < delta {
                        delta = slack[yy];
                        x = slackx[yy];
                        y = yy;
                    }
                }
                // If some slack has been found, remove it from x nodes in the
                // alternating path, and add it to y nodes in the alternating path.
                // The slack of y nodes outside the alternating path will be reduced
                // by this minimal slack as well.
                if delta > 0.0 {
                    for x in s.ones() {
                        lx[x] = lx[x] - delta;
                    }
                    for y in 0..ny {
                        if alternating[y].is_some() {
                            ly[y] = ly[y] + delta;
                        } else {
                            slack[y] = slack[y] - delta;
                        }
                    }
                }
                // Add (x, y) to the alternating path.
                alternating[y] = Some(x);
                if yx[y].is_none() {
                    // We have found an augmenting path.
                    break y;
                }
                // This y node had a predecessor, add it to the set of x nodes
                // in the augmenting path.
                let x = yx[y].unwrap();
                s.insert(x);
                // Update slack because of the added vertex in s might contain a
                // greater slack than with previously inserted x nodes in the augmenting
                // path.
                for y in 0..ny {
                    if alternating[y].is_none() {
                        let alternate_slack = lx[x] + ly[y] - copied_weights[(x, y)];
                        if slack[y] > alternate_slack.0 {
                            slack[y] = alternate_slack.0;
                            slackx[y] = x;
                        }
                    }
                }
            })
        };
        // Inverse edges along the augmenting path.
        while y.is_some() {
            let x = alternating[y.unwrap()].unwrap();
            let prec = xy[x];
            yx[y.unwrap()] = Some(x);
            xy[x] = y;
            y = prec;
        }
    }
    xy.into_iter()
        .enumerate()
        .map(|(i, v)| (i, v.unwrap()))
        .collect::<Vec<_>>()
}

#[pyclass(module = "fastmunk")]
struct FastMunk {
    maximize: bool,
}

#[pymethods]
impl FastMunk {
    #[new]
    #[pyo3(signature = (maximize = false))] // minimize by default
    fn new(maximize: bool) -> Self {
        FastMunk { maximize }
    }

    fn compute(&self, weights: Bound<'_, PyArray2<f64>>) -> PyResult<Vec<(usize, usize)>> {
        let weights = weights.readonly();
        let weights = weights.as_array();
        Ok(kuhn_munkres(weights, self.maximize))
    }
}

#[pymodule]
fn fastmunk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastMunk>()?;
    Ok(())
}
