//! # VAE Core Utilities
//!
//! This module contains shared constants, mathematical functions, and debugging utilities
//! used across both Convolutional and Dense Variational Autoencoder implementations.
//!
//! Key components:
//! * **Loss Calculation**: Implementation of the Evidence Lower Bound (ELBO).
//! * **Sampling**: The reparameterization trick for differentiable stochastic sampling.
//! * **Visualization Helpers**: Utilities for converting tensor data to image formats.
//! * **Introspection**: A module visitor for debugging model architecture and parameter shapes.

use burn::module::ModuleVisitor;
use burn::module::Param;
use burn::tensor::Distribution;
use burn::tensor::{Tensor, backend::Backend};

#[cfg(feature = "conv")]
pub mod conv_model;
#[cfg(feature = "dense")]
pub mod dense_model;
pub mod mnist_data;

// --- CONSTANTS ---

/// Width of MNIST images in pixels.
pub const MNIST_DIM_X: u32 = 28;

/// Height of MNIST images in pixels.
pub const MNIST_DIM_Y: u32 = 28;

/// Dimensionality of the VAE latent space (z).
///
/// This defines the size of the information bottleneck.
/// * A smaller dimension forces higher compression/abstraction.
/// * A larger dimension allows for better reconstruction but may lead to overfitting.
///
/// This constant is used globally for model configuration and CLI arguments.
pub const LATENT_DIM: usize = 4;

// --- LOSS FUNCTION ---

/// Computes the VAE Loss Function (Negative Evidence Lower Bound).
///
/// The loss is composed of two conflicting objectives:
/// 1. **Reconstruction Loss** (Binary Cross Entropy): Encourages the decoder to accurately reproduce the input.
/// 2. **Regularization** (KL Divergence): Forces the latent distribution `q(z|x)` to approximate a
///    standard normal distribution `N(0, 1)`.
///
/// # Formula
///
/// Loss = Loss_BCE + Loss_KL
///
/// Where:
///
/// * **Loss_BCE** = `-Σ [x · log(x̂) + (1−x) · log(1−x̂)]`
/// * **Loss_KL**  = `0.5 · Σ [exp(log(σ²)) + μ² - log(σ²) - 1]`
///
/// # Arguments
/// * `recon_x` - The reconstructed output (x̂) from the decoder. Shape: `(Batch, Input_Dim)`. Values in range `[0, 1]`.
/// * `x` - The original ground truth input. Shape: `(Batch, Input_Dim)`.
/// * `mu` - The predicted mean (μ) of the latent Gaussian. Shape: `(Batch, Latent_Dim)`.
/// * `sigma` - The predicted log-variance (log(σ²)) of the latent Gaussian. Shape: `(Batch, Latent_Dim)`.
///
/// # Returns
/// A scalar tensor representing the mean loss over the batch.
pub fn loss_function<B: Backend>(
    recon_x: Tensor<B, 2>,
    x: Tensor<B, 2>,
    mu: Tensor<B, 2>,
    sigma: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // Epsilon for numerical stability to prevent log(0) resulting in NaN/Inf.
    let eps = 1e-8;

    // Clamp predicted values to [eps, 1.0 - eps] before passing to log().
    let recon_clamp = recon_x.clamp(eps, 1.0 - eps);

    // Compute Binary Cross-Entropy (BCE)
    // Formula: -( x * log(x̂) + (1 - x) * log(1 - x̂) )
    let bce = (x.clone() * recon_clamp.clone().log()
        + x.clone().neg().add_scalar(1.0) * recon_clamp.neg().add_scalar(1.0).log())
    .neg();

    // Reconstruction Loss: Sum over features, Mean over batch.
    let recon_loss = bce.sum_dim(1).mean();

    // Compute Kullback-Leibler (KL) Divergence analytically.
    // Measures the difference between the learned distribution and N(0, 1).
    // Formula: 0.5 * sum( exp(logvar) + mu^2 - logvar - 1 )
    let kld = (sigma.clone().exp() + mu.powf_scalar(2.0) - sigma - 1.0)
        .sum_dim(1)
        .mean() // Mean over batch
        .mul_scalar(0.5);

    // Total Loss = Reconstruction + Regularization
    recon_loss + kld
}

/// Converts normalized floating-point pixel data into raw RGB bytes.
///
/// Used for visualization tools that require `u8` buffers.
///
/// # Arguments
/// * `pixels` - A slice of floating point values in range `[0.0, 1.0]`.
///
/// # Returns
/// A vector of bytes where every grayscale pixel is expanded to three RGB bytes.
/// Example: `0.5` -> `[127, 127, 127]`.
pub fn build_rgb_bytes(pixels: &[f32]) -> Vec<u8> {
    let mut res = Vec::with_capacity(pixels.len() * 3);
    for &val in pixels {
        let u = (val * 255.0).clamp(0.0, 255.0) as u8;
        res.push(u); // R
        res.push(u); // G
        res.push(u); // B
    }
    res
}

/// Implements the Reparameterization Trick.
///
/// Allows backpropagation to flow through the stochastic sampling node in the network.
/// Instead of sampling `z ~ N(μ, σ²)` directly (which is non-differentiable),
/// we sample noise `ε ~ N(0, 1)` and compute:
///
/// z = μ + σ · ε
///
/// # Arguments
/// * `mu` - The mean vector (μ).
/// * `logvar` - The log-variance vector (log(σ²)).
///
/// # Returns
/// A sampled latent vector `z` compatible with backpropagation.
pub fn reparameterize<B: Backend, const D: usize>(
    mu: Tensor<B, D>,
    logvar: Tensor<B, D>,
) -> Tensor<B, D> {
    // Convert log-variance to standard deviation: σ = exp(0.5 * logvar)
    let std = logvar.mul_scalar(0.5).exp();

    // Sample epsilon (noise) from Standard Normal Distribution N(0, 1)
    let eps = Tensor::random_like(&std, Distribution::Normal(0.0, 1.0));

    // Scale and shift the noise
    mu + eps * std
}

// --- MODEL INSPECTION ---

/// A debugging utility for inspecting Burn model architectures.
///
/// Implements the `ModuleVisitor` trait to traverse the model tree recursively,
/// printing module names and parameter shapes (e.g., Weights, Biases).
pub struct ModelTreePrinter {
    indent: usize,
}

impl ModelTreePrinter {
    /// Creates a new printer starting at indentation level 0.
    pub fn new() -> Self {
        Self { indent: 0 }
    }

    /// Helper to print current indentation.
    fn print_indent(&self) {
        print!("{}", "  ".repeat(self.indent));
    }
}

impl<B: Backend> ModuleVisitor<B> for ModelTreePrinter {
    /// Called when the visitor enters a child module.
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.print_indent();
        println!("Module: {}", name);
        self.indent += 1;
    }

    /// Called when the visitor finishes a module.
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.indent -= 1;
    }

    /// Called for floating-point parameters (Weights, Biases).
    /// Prints the parameter shape and unique ID.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.print_indent();
        println!("Param (Float): {:?} [ID: {}]", param.shape(), param.id);
    }

    /// Called for integer parameters.
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, burn::tensor::Int>>) {
        self.print_indent();
        println!("Param (Int): {:?}", param.shape());
    }

    /// Called for boolean parameters.
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, burn::tensor::Bool>>) {
        self.print_indent();
        println!("Param (Bool): {:?}", param.shape());
    }
}
