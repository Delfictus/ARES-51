//! Neural network implementation for ADP

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Simple feedforward neural network
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

struct Layer {
    weights: Array2<f64>,
    bias: Array1<f64>,
    activation: Activation,
}

#[derive(Clone, Copy)]
enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(anyhow::anyhow!("Need at least input and output layers"));
        }

        let mut layers = Vec::new();
        let mut rng = rand::thread_rng();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / input_size as f64).sqrt();
            let weights =
                Array2::from_shape_fn((output_size, input_size), |_| rng.gen_range(-scale..scale));

            let bias = Array1::zeros(output_size);

            // Use ReLU for hidden layers, linear for output
            let activation = if i == layer_sizes.len() - 2 {
                Activation::Linear
            } else {
                Activation::ReLU
            };

            layers.push(Layer {
                weights,
                bias,
                activation,
            });
        }

        Ok(Self {
            layers,
            learning_rate,
        })
    }

    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut output = input.clone();

        for layer in &self.layers {
            // Linear transformation
            output = layer.weights.dot(&output) + &layer.bias;

            // Vectorized activation functions
            output.mapv_inplace(|x| match layer.activation {
                Activation::ReLU => x.max(0.0),
                Activation::Tanh => x.tanh(),
                Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Activation::Linear => x,
            });
        }

        Ok(output)
    }

    pub fn backward(&mut self, input: &Array1<f64>, target: &Array1<f64>) -> Result<()> {
        // Simple gradient descent - in production would use proper backprop
        let output = self.forward(input)?;
        let error = &output - target;

        // Update last layer (simplified)
        if let Some(last_layer) = self.layers.last_mut() {
            let gradient = error.clone() * self.learning_rate;
            last_layer.bias = &last_layer.bias - &gradient;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network() {
        let nn = NeuralNetwork::new(&[10, 5, 3], 0.01).unwrap();
        let input = Array1::from_vec(vec![0.5; 10]);
        let output = nn.forward(&input).unwrap();
        assert_eq!(output.len(), 3);
    }
}
