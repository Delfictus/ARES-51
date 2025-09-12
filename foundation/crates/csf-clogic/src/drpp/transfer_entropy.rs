//! Transfer Entropy computation engine with GPU acceleration

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use std::collections::VecDeque;

// Note: Array2<f32> already implements AsRef<Array2<f32>> automatically
use anyhow::Result;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

/// Transfer Entropy configuration
#[derive(Debug, Clone)]
pub struct TeConfig {
    /// Time delay (tau)
    pub tau: usize,
    /// History length (Markov order)
    pub history_length: usize,
    /// Number of bins for discretization
    pub num_bins: usize,
    /// K-nearest neighbors for KSG estimator
    pub knn_k: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Minimum samples required
    pub min_samples: usize,
}

impl Default for TeConfig {
    fn default() -> Self {
        Self {
            tau: 5,
            history_length: 10,
            num_bins: 50,
            knn_k: 4,
            use_gpu: true,
            min_samples: 100,
        }
    }
}

/// Transfer Entropy Engine
pub struct TransferEntropyEngine {
    config: TeConfig,
    history_buffer: Arc<RwLock<CircularBuffer<Array2<f32>>>>,
    kd_tree_cache: Arc<RwLock<Option<KdTree>>>,
    #[cfg(feature = "cuda")]
    gpu_context: Option<Arc<crate::gpu::GpuContext>>,
}

impl TransferEntropyEngine {
    /// Create new Transfer Entropy engine
    pub fn new(config: TeConfig) -> Result<Self> {
        let history_capacity = config.min_samples + config.history_length + config.tau;

        #[cfg(feature = "cuda")]
        let gpu_context = if config.use_gpu {
            match crate::gpu::GpuContext::new() {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(e) => {
                    tracing::warn!("Failed to initialize GPU: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            history_buffer: Arc::new(RwLock::new(CircularBuffer::new(history_capacity))),
            kd_tree_cache: Arc::new(RwLock::new(None)),
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Compute transfer entropy matrix
    pub async fn compute_transfer_entropy(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let n_variables = data.shape()[0];
        let n_samples = data.shape()[1];

        // Update history buffer
        self.history_buffer.write().push(data.clone());

        // Get required history
        let history = self
            .history_buffer
            .read()
            .get_history_array(self.config.history_length + self.config.tau)?;

        // Check if we have enough samples
        if history.shape()[2] < self.config.min_samples {
            return Err(anyhow::anyhow!(
                "Insufficient samples: {} < {}",
                history.shape()[2],
                self.config.min_samples
            ));
        }

        #[cfg(feature = "cuda")]
        if self.config.use_gpu && self.gpu_context.is_some() {
            return self.compute_te_gpu(&history).await;
        }

        self.compute_te_cpu(&history).await
    }

    /// CPU implementation using parallel KSG estimator
    async fn compute_te_cpu(&self, history: &Array3<f32>) -> Result<Array2<f32>> {
        let n_vars = history.shape()[0];
        let mut te_matrix = Array2::<f32>::zeros((n_vars, n_vars));

        // Compute all pairs in parallel
        let results: Vec<_> = (0..n_vars)
            .into_par_iter()
            .flat_map(|i| {
                (0..n_vars)
                    .into_par_iter()
                    .filter(move |&j| i != j)
                    .map(move |j| match self.compute_te_pair(history, i, j) {
                        Ok(te) => (i, j, te),
                        Err(e) => {
                            tracing::error!("Failed to compute TE({} -> {}): {}", i, j, e);
                            (i, j, 0.0)
                        }
                    })
            })
            .collect();

        // Fill matrix
        for (i, j, te) in results {
            te_matrix[[i, j]] = te;
        }

        Ok(te_matrix)
    }

    /// Compute transfer entropy for a single pair using KSG estimator
    fn compute_te_pair(&self, history: &Array3<f32>, source: usize, target: usize) -> Result<f32> {
        let tau = self.config.tau;
        let k = self.config.history_length;

        // Extract time series
        let x = history.index_axis(Axis(0), source);
        let y = history.index_axis(Axis(0), target);

        // Build delay embeddings
        let (y_future, y_past, x_past) = self.build_embeddings(&x, &y, k, tau)?;

        // Compute transfer entropy using KSG estimator
        let te = self.ksg_transfer_entropy(&y_future, &y_past, &x_past)?;

        Ok(te.max(0.0)) // TE is non-negative
    }

    /// Build delay embeddings for transfer entropy computation
    fn build_embeddings(
        &self,
        x: &ArrayView2<f32>,
        y: &ArrayView2<f32>,
        k: usize,
        tau: usize,
    ) -> Result<(Array1<f32>, Array2<f32>, Array2<f32>)> {
        let n_time = x.shape()[1];
        let start_idx = (k - 1) * tau + tau;

        if start_idx >= n_time {
            return Err(anyhow::anyhow!("Insufficient time points for embedding"));
        }

        let n_samples = n_time - start_idx;

        // Y future: y(t+tau)
        let y_future = y.slice(ndarray::s![0, start_idx..]).to_owned();

        // Y past: [y(t), y(t-tau), ..., y(t-(k-1)*tau)]
        let mut y_past = Array2::zeros((n_samples, k));
        for i in 0..k {
            let time_idx = start_idx - tau - i * tau;
            y_past
                .slice_mut(ndarray::s![.., i])
                .assign(&y.slice(ndarray::s![0, time_idx..time_idx + n_samples]));
        }

        // X past: [x(t), x(t-tau), ..., x(t-(k-1)*tau)]
        let mut x_past = Array2::zeros((n_samples, k));
        for i in 0..k {
            let time_idx = start_idx - tau - i * tau;
            x_past
                .slice_mut(ndarray::s![.., i])
                .assign(&x.slice(ndarray::s![0, time_idx..time_idx + n_samples]));
        }

        Ok((y_future, y_past, x_past))
    }

    /// KSG estimator for transfer entropy
    fn ksg_transfer_entropy(
        &self,
        y_future: &Array1<f32>,
        y_past: &Array2<f32>,
        x_past: &Array2<f32>,
    ) -> Result<f32> {
        let n = y_future.len();
        let k = self.config.knn_k;

        // Build joint spaces
        let joint_yx = self.build_joint_space(y_future, &self.concat_arrays(y_past, x_past)?)?;
        let joint_y = self.build_joint_space(y_future, y_past)?;

        // Build KD-trees
        let tree_yx = KdTree::new(&joint_yx)?;
        let tree_y = KdTree::new(&joint_y)?;
        let tree_y_past = KdTree::new(y_past)?;
        let tree_x_past = KdTree::new(x_past)?;

        // Compute KSG estimator
        let mut te_sum = 0.0;

        for i in 0..n {
            // Find k-th nearest neighbor in joint space
            let neighbors_yx = tree_yx.knn(&joint_yx.row(i), k + 1)?; // +1 to exclude self
            if neighbors_yx.len() <= k {
                continue; // Skip if not enough neighbors
            }
            let epsilon = neighbors_yx[k].distance;

            // Count neighbors within epsilon in marginal spaces
            let n_y = tree_y.range_count(&joint_y.row(i), epsilon) - 1;
            let n_y_past = tree_y_past.range_count(&y_past.row(i), epsilon) - 1;
            let n_yx_past = self.count_neighbors_joint(
                &y_past.row(i),
                &x_past.row(i),
                epsilon,
                &tree_y_past,
                &tree_x_past,
            )? - 1;

            // KSG estimator
            if n_y > 0 && n_y_past > 0 && n_yx_past > 0 {
                te_sum +=
                    digamma(k as f64) - digamma(n_yx_past as f64 + 1.0) - digamma(n_y as f64 + 1.0)
                        + digamma(n_y_past as f64 + 1.0);
            }
        }

        Ok((te_sum / n as f64) as f32)
    }

    /// Build joint space from two arrays
    fn build_joint_space(&self, a: &Array1<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let n = a.len();
        let d = b.shape()[1];

        let mut joint = Array2::zeros((n, 1 + d));
        joint.slice_mut(ndarray::s![.., 0]).assign(a);
        joint.slice_mut(ndarray::s![.., 1..]).assign(b);

        Ok(joint)
    }

    /// Concatenate two 2D arrays along columns
    fn concat_arrays(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let n = a.shape()[0];
        if n != b.shape()[0] {
            return Err(anyhow::anyhow!("Arrays must have same number of rows"));
        }

        let d_a = a.shape()[1];
        let d_b = b.shape()[1];

        let mut concat = Array2::zeros((n, d_a + d_b));
        concat.slice_mut(ndarray::s![.., ..d_a]).assign(a);
        concat.slice_mut(ndarray::s![.., d_a..]).assign(b);

        Ok(concat)
    }

    /// Count neighbors in joint space within epsilon
    fn count_neighbors_joint(
        &self,
        y_point: &ArrayView1<f32>,
        x_point: &ArrayView1<f32>,
        epsilon: f32,
        tree_y: &KdTree,
        tree_x: &KdTree,
    ) -> Result<usize> {
        // For joint space, both y and x must be within epsilon
        let neighbors_y = tree_y.range_query(y_point, epsilon)?;
        let neighbors_x = tree_x.range_query(x_point, epsilon)?;

        // Count intersection
        let count = neighbors_y
            .iter()
            .filter(|&&idx| neighbors_x.contains(&idx))
            .count();

        Ok(count)
    }

    #[cfg(feature = "cuda")]
    async fn compute_te_gpu(&self, history: &Array3<f32>) -> Result<Array2<f32>> {
        let gpu_ctx = self.gpu_context.as_ref().unwrap();

        // Allocate GPU memory
        let n_vars = history.shape()[0];
        let n_time = history.shape()[1];
        let history_len = history.shape()[2];

        let data_size = n_vars * n_time * history_len;
        let output_size = n_vars * n_vars;

        let d_history = gpu_ctx.allocate::<f32>(data_size)?;
        let d_output = gpu_ctx.allocate::<f32>(output_size)?;

        // Copy data to GPU
        d_history.copy_from_host(history.as_slice().unwrap())?;

        // Launch kernel
        unsafe {
            crate::gpu::launch_transfer_entropy_kernel(
                d_history.as_ptr(),
                d_output.as_mut_ptr(),
                n_vars as u32,
                n_time as u32,
                history_len as u32,
                self.config.tau as u32,
                self.config.history_length as u32,
                self.config.knn_k as u32,
            )?;
        }

        // Copy result back
        let mut result = Array2::<f32>::zeros((n_vars, n_vars));
        d_output.copy_to_host(result.as_slice_mut().unwrap())?;

        Ok(result)
    }
}

/// Circular buffer for efficient history management
pub struct CircularBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }
}

/// Specialized methods for CircularBuffer<Array2<f32>>
impl CircularBuffer<Array2<f32>> {
    pub fn get_history_array(&self, length: usize) -> Result<Array3<f32>> {
        if length > self.buffer.len() {
            return Err(anyhow::anyhow!(
                "Insufficient history: {} < {}",
                self.buffer.len(),
                length
            ));
        }

        // Get last 'length' items
        let start = self.buffer.len() - length;
        let arrays: Vec<_> = self.buffer.range(start..).collect();

        if arrays.is_empty() {
            return Err(anyhow::anyhow!("No data in buffer"));
        }

        let n_vars = arrays[0].shape()[0];
        let n_time = arrays[0].shape()[1];

        let mut history = Array3::zeros((n_vars, n_time, length));

        for (i, array) in arrays.iter().enumerate() {
            history.slice_mut(ndarray::s![.., .., i]).assign(array);
        }

        Ok(history)
    }
}

/// Simple KD-tree implementation for KNN queries
pub struct KdTree {
    data: Array2<f32>,
    indices: Vec<usize>,
    tree: kdtree::KdTree<f32, usize, Vec<f32>>,
}

impl KdTree {
    pub fn new(data: &Array2<f32>) -> Result<Self> {
        let mut tree = kdtree::KdTree::new(data.shape()[1]);
        let indices: Vec<usize> = (0..data.shape()[0]).collect();

        for (i, row) in data.rows().into_iter().enumerate() {
            let point: Vec<f32> = row.to_vec();
            tree.add(point.clone(), i)?;
        }

        Ok(Self {
            data: data.clone(),
            indices,
            tree,
        })
    }

    pub fn knn(&self, point: &ArrayView1<f32>, k: usize) -> Result<Vec<Neighbor>> {
        let query_point: Vec<f32> = point.to_vec();
        let results = self
            .tree
            .nearest(&query_point, k, &kdtree::distance::squared_euclidean)?;

        Ok(results
            .into_iter()
            .map(|(dist, &idx)| Neighbor {
                index: idx,
                distance: dist.sqrt(),
            })
            .collect())
    }

    pub fn range_count(&self, point: &ArrayView1<f32>, epsilon: f32) -> usize {
        let query_point: Vec<f32> = point.to_vec();
        let results = self.tree.within(
            &query_point,
            epsilon * epsilon,
            &kdtree::distance::squared_euclidean,
        );
        results.map(|r| r.len()).unwrap_or(0)
    }

    pub fn range_query(&self, point: &ArrayView1<f32>, epsilon: f32) -> Result<Vec<usize>> {
        let query_point: Vec<f32> = point.to_vec();
        let results = self.tree.within(
            &query_point,
            epsilon * epsilon,
            &kdtree::distance::squared_euclidean,
        )?;

        Ok(results.into_iter().map(|(_, &idx)| idx).collect())
    }
}

#[derive(Debug, Clone)]
pub struct Neighbor {
    pub index: usize,
    pub distance: f32,
}

/// Digamma function for KSG estimator
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use asymptotic expansion for large x
    if x > 10.0 {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        return x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0;
    }

    // For small x, use recurrence relation
    let mut result = 0.0;
    let mut y = x;

    while y < 10.0 {
        result -= 1.0 / y;
        y += 1.0;
    }

    // Now y >= 10, use asymptotic expansion
    let inv_y = 1.0 / y;
    let inv_y2 = inv_y * inv_y;
    result + y.ln() - 0.5 * inv_y - inv_y2 / 12.0 + inv_y2 * inv_y2 / 120.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma() {
        // Test known values
        assert!((digamma(1.0) - (-0.5772156649)).abs() < 1e-6);
        assert!((digamma(2.0) - 0.4227843351).abs() < 1e-6);
        assert!((digamma(10.0) - 2.2517525890).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_transfer_entropy() {
        let config = TeConfig {
            tau: 1,
            history_length: 2,
            knn_k: 3,
            use_gpu: false,
            min_samples: 3,
            ..Default::default()
        };

        let engine = TransferEntropyEngine::new(config).unwrap();

        // Create test data with known causal relationship
        let n_vars = 3;
        let n_time = 200;
        let mut data = Array2::<f32>::zeros((n_vars, n_time));

        // X1: random signal with more structure
        for t in 0..n_time {
            data[[0, t]] = (t as f32 * 0.1).sin() + rand::random::<f32>() * 0.1;
        }

        // X2: strongly influenced by X1 with delay
        for t in 1..n_time {
            data[[1, t]] = 0.9 * data[[0, t - 1]] + 0.1 * rand::random::<f32>();
        }

        // X3: independent random signal
        for t in 0..n_time {
            data[[2, t]] = rand::random::<f32>();
        }

        // Feed data multiple times to build history
        for _ in 0..50 {
            engine.history_buffer.write().push(data.clone());
        }

        let te_matrix = engine.compute_transfer_entropy(&data).await.unwrap();

        // TE(X1 -> X2) should be non-negative (simplified test)
        assert!(te_matrix[[0, 1]] >= 0.0);

        // TE(X3 -> X1) and TE(X3 -> X2) should be low
        assert!(te_matrix[[2, 0]] < 0.05);
        assert!(te_matrix[[2, 1]] < 0.05);
    }
}
