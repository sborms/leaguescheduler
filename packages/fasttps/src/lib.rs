use std::collections::{HashMap, HashSet};

use fixedbitset::FixedBitSet;
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ─── Constants (must match Python constants.py) ──────────────────────────────
const DISALLOWED_NBR: f64 = 9_999_999.0;
const LARGE_NBR: f64 = 9999.0;
const DISALLOWED_REPLACE: f64 = 1e15;

// ─── Hungarian algorithm (Kuhn-Munkres) ──────────────────────────────────────
// Courtesy to https://github.com/neka-nat/fastmunk for main implementation
fn kuhn_munkres(weights: ArrayView2<f64>, maximize: bool) -> Vec<(usize, usize)> {
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
    let mut xy: Vec<Option<usize>> = vec![None; nx];
    let mut yx: Vec<Option<usize>> = vec![None; ny];
    let mut lx: Vec<OrderedFloat<f64>> = (0..nx)
        .map(|row| {
            (0..ny)
                .map(|col| OrderedFloat(copied_weights[(row, col)]))
                .max()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut ly: Vec<f64> = vec![0.0; ny];
    let mut s = FixedBitSet::with_capacity(nx);
    let mut alternating = Vec::with_capacity(ny);
    let mut slack = vec![0.0; ny];
    let mut slackx = Vec::with_capacity(ny);
    for root in 0..nx {
        alternating.clear();
        alternating.resize(ny, None);
        let mut y = {
            s.clear();
            s.insert(root);
            for y in 0..ny {
                slack[y] = lx[root].0 + ly[y] - copied_weights[(root, y)];
            }
            slackx.clear();
            slackx.resize(ny, root);
            Some(loop {
                let mut delta = f64::INFINITY;
                let mut x = 0;
                let mut y = 0;
                for yy in 0..ny {
                    if alternating[yy].is_none() && slack[yy] < delta {
                        delta = slack[yy];
                        x = slackx[yy];
                        y = yy;
                    }
                }
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
                alternating[y] = Some(x);
                if yx[y].is_none() {
                    break y;
                }
                let x = yx[y].unwrap();
                s.insert(x);
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

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Equivalent of Python _get_team_array: concatenate X[idx,:] and X[:,idx], filter out LARGE_NBR.
#[inline]
fn get_team_array(x: &ArrayView2<f64>, idx: usize) -> Vec<f64> {
    let n = x.nrows();
    let mut arr = Vec::with_capacity(2 * n);
    for j in 0..x.ncols() {
        let v = x[(idx, j)];
        if v != LARGE_NBR {
            arr.push(v);
        }
    }
    for i in 0..n {
        let v = x[(i, idx)];
        if v != LARGE_NBR {
            arr.push(v);
        }
    }
    arr
}

/// Minimum positive value in a slice of (value - h) differences.
/// Returns None if no positive value exists.
#[inline]
fn min_positive_delta(games: &[f64], h: f64, forward: bool) -> Option<f64> {
    let mut best = f64::INFINITY;
    for &g in games {
        let d = if forward { g - h } else { h - g };
        if d > 0.0 && d < best {
            best = d;
        }
    }
    if best == f64::INFINITY {
        None
    } else {
        Some(best)
    }
}

/// Build cost matrix for a given team. This is the #1 bottleneck.
fn create_cost_matrix_inner(
    x: &ArrayView2<f64>,
    team_idx: usize,
    set_home: &[f64],
    opponents: &[usize],
    sets_forbidden: &HashMap<usize, HashSet<i64>>,
    m: f64,
    r_max: f64,
    penalties: &[i64],
    max_penalty_key: usize,
) -> Array2<f64> {
    let n_home = set_home.len();
    let n_oppo = opponents.len();
    let mut am_cost = Array2::<f64>::zeros((n_home, n_oppo));

    // precompute games for the home team
    let games_team = get_team_array(x, team_idx);

    // C2 - home date availability (home_dates = set_home)

    // C4 (team part) - precompute which home dates the team already plays on
    // C5 (team part) - precompute whether any game is within r_max of each home date
    let mut team_plays_mask = vec![false; n_home];
    let mut games_in_r_max_team = vec![false; n_home];

    for (i, &h) in set_home.iter().enumerate() {
        for &g in &games_team {
            // C4 - team already plays game
            if g == h {
                team_plays_mask[i] = true;
            }
            // C5 - distance < r_max means too close
            let d = (g - h).abs() + 1.0;
            if d < r_max {
                games_in_r_max_team[i] = true;
            }
        }
    }

    for (j, &oppo_idx) in opponents.iter().enumerate() {
        let games_oppo = get_team_array(x, oppo_idx);

        // get forbidden set for this opponent (empty if not present)
        let forbidden_set = sets_forbidden.get(&oppo_idx);

        // C6 - reciprocal game value (game j->i)
        let reciprocal_val = x[(oppo_idx, team_idx)];

        for (i, &h) in set_home.iter().enumerate() {
            let h_i64 = h as i64;

            // C3 - forbidden game set
            let forbidden = match forbidden_set {
                Some(fs) => fs.contains(&h_i64),
                None => false,
            };

            // C4 - team already plays game (precomputed)
            let team_plays = team_plays_mask[i];

            // C4 - opponent already plays game
            let mut oppo_plays = false;
            for &g in &games_oppo {
                if g == h {
                    oppo_plays = true;
                    break;
                }
            }

            // C5 - max. 2 games for 'r_max' slots
            let mut games_in_r_max_oppo = false;
            for &g in &games_oppo {
                let d = (g - h).abs() + 1.0;
                if d < r_max {
                    games_in_r_max_oppo = true;
                    break;
                }
            }
            let r_max_violated = games_in_r_max_team[i] || games_in_r_max_oppo;

            // C6 - game i-j is within m days of game j-i
            let reciprocal_too_close = (h - reciprocal_val).abs() < m;

            // set disallowed cost for all disallowed slots
            let disallowed =
                forbidden || team_plays || oppo_plays || reciprocal_too_close || r_max_violated;

            if disallowed {
                am_cost[(i, j)] = DISALLOWED_NBR;
            } else {
                // process penalty calculations for allowed slots
                let mut total_penalty: i64 = 0;

                // forward-looking for team
                if let Some(d) = min_positive_delta(&games_team, h, true) {
                    let key = d as usize;
                    if key <= max_penalty_key {
                        total_penalty += penalties[key];
                    }
                }
                // backward-looking for team
                if let Some(d) = min_positive_delta(&games_team, h, false) {
                    let key = d as usize;
                    if key <= max_penalty_key {
                        total_penalty += penalties[key];
                    }
                }
                // forward-looking for opponent
                if let Some(d) = min_positive_delta(&games_oppo, h, true) {
                    let key = d as usize;
                    if key <= max_penalty_key {
                        total_penalty += penalties[key];
                    }
                }
                // backward-looking for opponent
                if let Some(d) = min_positive_delta(&games_oppo, h, false) {
                    let key = d as usize;
                    if key <= max_penalty_key {
                        total_penalty += penalties[key];
                    }
                }

                am_cost[(i, j)] = total_penalty as f64;
            }
        }
    }

    am_cost
}

/// Full solve: cost matrix → adjacency matrix → Munkres → process indexes → (pick, total_cost).
fn solve_inner(
    x: &ArrayView2<f64>,
    team_idx: usize,
    set_home: &[f64],
    opponents: &[usize],
    sets_forbidden: &HashMap<usize, HashSet<i64>>,
    m: f64,
    p: f64,
    r_max: f64,
    penalties: &[i64],
    max_penalty_key: usize,
) -> (Vec<f64>, f64) {
    let n_home = set_home.len();
    let n_oppo = opponents.len();
    let dim = n_home + n_oppo;

    // construct cost matrix
    let am_cost =
        create_cost_matrix_inner(x, team_idx, set_home, opponents, sets_forbidden, m, r_max, penalties, max_penalty_key);

    // construct full adjacency matrix:
    // [ am_cost  |  zeros ]
    // [ am_bott  |  zeros ]
    // where am_bott is n_oppo x n_oppo filled with p
    // and zeros is dim x (dim - n_oppo)
    let mut am = Array2::<f64>::zeros((dim, dim));

    // fill cost block (top-left: n_home x n_oppo)
    for i in 0..n_home {
        for j in 0..n_oppo {
            let v = am_cost[(i, j)];
            // replace disallowed values with a large number
            am[(i, j)] = if v == DISALLOWED_NBR { DISALLOWED_REPLACE } else { v };
        }
    }

    // fill bottom block (n_oppo x n_oppo) with p
    for i in 0..n_oppo {
        for j in 0..n_oppo {
            am[(n_home + i, j)] = p;
        }
    }
    // right block stays zeros (already initialized)

    // run Hungarian algorithm (minimize)
    let indexes = kuhn_munkres(am.view(), false);

    // compute total cost
    let total_cost: f64 = indexes.iter().map(|&(r, c)| am[(r, c)]).sum();

    // process optimal indexes: sort by opponent, map to home slot or NaN
    let mut indexes_inv: Vec<(usize, f64)> = Vec::with_capacity(n_oppo);
    for &(k, v) in &indexes {
        if v < n_oppo {
            let slot = if k < n_home {
                set_home[k]
            } else {
                f64::NAN
            };
            indexes_inv.push((opponents[v], slot));
        }
    }
    indexes_inv.sort_by_key(|&(opp, _)| opp);

    let pick: Vec<f64> = indexes_inv.iter().map(|&(_, slot)| slot).collect();

    (pick, total_cost)
}

// ─── PyO3 class ──────────────────────────────────────────────────────────────

#[pyclass(module = "fasttps")]
struct FastTPS {
    sets_home: HashMap<usize, Vec<f64>>,
    sets_forbidden: HashMap<usize, HashSet<i64>>,
    m: f64,
    p: f64,
    r_max: f64,
    /// Flat penalty lookup: penalties_vec[d] = penalty for distance d.
    penalties_vec: Vec<i64>,
    max_penalty_key: usize,
}

#[pymethods]
impl FastTPS {
    #[new]
    #[pyo3(signature = (sets_home, sets_forbidden, m, p, r_max, penalties))]
    fn new(
        sets_home: &Bound<'_, PyDict>,
        sets_forbidden: &Bound<'_, PyDict>,
        m: i64,
        p: i64,
        r_max: i64,
        penalties: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // parse sets_home: dict[int, list/set of float/int]
        let mut sh: HashMap<usize, Vec<f64>> = HashMap::new();
        for (k, v) in sets_home.iter() {
            let key: usize = k.extract()?;
            let vals: Vec<f64> = v.extract()?;
            sh.insert(key, vals);
        }

        // parse sets_forbidden: dict[int, list/set of float/int] → HashSet<i64>
        let mut sf: HashMap<usize, HashSet<i64>> = HashMap::new();
        for (k, v) in sets_forbidden.iter() {
            let key: usize = k.extract()?;
            let vals: Vec<i64> = v.extract()?;
            sf.insert(key, vals.into_iter().collect());
        }

        // parse penalties dict to flat Vec
        let mut max_key: usize = 0;
        let mut pen_map: HashMap<usize, i64> = HashMap::new();
        for (k, v) in penalties.iter() {
            let key: usize = k.extract()?;
            let val: i64 = v.extract()?;
            pen_map.insert(key, val);
            if key > max_key {
                max_key = key;
            }
        }
        let mut penalties_vec = vec![0i64; max_key + 1];
        for (k, v) in &pen_map {
            penalties_vec[*k] = *v;
        }

        Ok(FastTPS {
            sets_home: sh,
            sets_forbidden: sf,
            m: m as f64,
            p: p as f64,
            r_max: r_max.max(2) as f64,
            penalties_vec,
            max_penalty_key: max_key,
        })
    }

    /// Solve transportation problem for given home team.
    /// Modifies X in-place and returns total_cost.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        x_py: &Bound<'py, PyArray2<f64>>,
        team_idx: usize,
    ) -> PyResult<f64> {
        let x_read = x_py.readonly();
        let x_view = x_read.as_array();
        let n = x_view.nrows();

        let set_home = self
            .sets_home
            .get(&team_idx)
            .expect("team_idx not in sets_home");
        let opponents: Vec<usize> = (0..n).filter(|&t| t != team_idx).collect();

        let (pick, total_cost) = solve_inner(
            &x_view,
            team_idx,
            set_home,
            &opponents,
            &self.sets_forbidden,
            self.m,
            self.p,
            self.r_max,
            &self.penalties_vec,
            self.max_penalty_key,
        );

        // release readonly borrow before writing
        drop(x_read);

        // assign selection to X in-place: X[team_idx, opponents] = pick
        unsafe {
            let mut x_mut = x_py.as_array_mut();
            for (idx, &opp) in opponents.iter().enumerate() {
                *x_mut.uget_mut([team_idx, opp]) = pick[idx];
            }
        }

        Ok(total_cost)
    }

    /// Create cost matrix (exposed for construction phase method 2).
    fn create_cost_matrix<'py>(
        &self,
        py: Python<'py>,
        x_py: PyReadonlyArray2<'py, f64>,
        team_idx: usize,
        set_home: Vec<f64>,
        opponents: Vec<usize>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x = x_py.as_array();
        let am = create_cost_matrix_inner(
            &x,
            team_idx,
            &set_home,
            &opponents,
            &self.sets_forbidden,
            self.m,
            self.r_max,
            &self.penalties_vec,
            self.max_penalty_key,
        );
        Ok(PyArray2::from_owned_array_bound(py, am).unbind())
    }
}

#[pymodule]
fn fasttps(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastTPS>()?;
    Ok(())
}
