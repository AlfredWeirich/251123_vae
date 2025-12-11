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
    logvar: Tensor<B, 2>,
    beta: f32,
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
    // println!(
    //     "Reconstruction Loss: {:?}",
    //     recon_loss.to_data().to_vec::<f32>()
    // );

    // Compute Kullback-Leibler (KL) Divergence analytically.
    // Measures the difference between the learned distribution and N(0, 1).
    // Formula: 0.5 * sum( exp(logvar) + mu^2 - logvar - 1 )
    let kld = (logvar.clone().exp() + mu.powf_scalar(2.0) - logvar - 1.0)
        .sum_dim(1)
        .mean() // Mean over batch
        .mul_scalar(0.5);
    // println!("KL Divergence: {:?}", kld.to_data().to_vec::<f32>());

    // Total Loss = Reconstruction + Regularization * beta
    recon_loss + kld.mul_scalar(beta)
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
impl Default for ModelTreePrinter {
    fn default() -> Self {
        Self::new()
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

/// Computes the dynamic KL Divergence weight ($\beta$) using a Cyclic/Linear Annealing schedule.
///
/// This technique helps prevent "Posterior Collapse" (where the decoder ignores the latent space)
/// by letting the model learn to reconstruct images as a pure Autoencoder first, before
/// slowly introducing the Gaussian regularization constraint.
///
/// # Schedule Strategy
/// The training is divided into three distinct phases based on the `progress` ($t$):
///
/// 1. **Warmup Phase (0% - 20%)**: $\beta = 0.0$.
///    The model trains purely on reconstruction loss. The latent space organizes itself freely
///    without being forced into a Gaussian shape.
///
/// 2. **Annealing Phase (20% - 70%)**: $\beta \to 0.0 \dots 1.0$.
///    The regularization weight increases linearly. The clusters formed in Phase 1 are
///    gently pulled towards the center to approximate $N(0,1)$.
///
/// 3. **Standard Phase (70% - 100%)**: $\beta = 1.0$.
///    Standard VAE training. The latent space is fully regularized to ensure continuity
///    for sampling.
///
/// # Arguments
///
/// * `current_epoch` - The current training epoch (0-indexed).
/// * `max_epochs` - The total number of epochs defined for training.
///
/// # Returns
///
/// A `f32` value between `0.0` and `1.0`.
pub fn calculate_kl_beta_ramp(current_epoch: usize, max_epochs: usize, max_beta: f32) -> f32 {
    // Edge case: Prevent division by zero if training configuration is invalid.
    if max_epochs == 0 {
        return 1.0;
    }

    // Normalized progress ratio [0.0, 1.0] (or > 1.0 if over-training).
    let progress = current_epoch as f32 / max_epochs as f32;

    if progress < 0.2 {
        // --- PHASE 1: Deterministic Warmup ---
        // Disable KL loss completely.
        // Goal: Minimize Reconstruction Error (MSE/BCE) only.
        0.0
    } else if progress < 0.7 {
        // --- PHASE 2: Linear Annealing ---
        // Linearly interpolate Beta from 0.0 to 1.0 over 50% of the training duration.
        // Formula: (Current_Progress - Phase_Start) / Phase_Duration
        // Range: (0.2 -> 0.7) maps to (0.0 -> 1.0)
        (progress - 0.2) / 0.5 * max_beta
    } else {
        // --- PHASE 3: Standard VAE ---
        // Full regularization enabled.
        // Goal: Ensure latent space is valid for generation (Standard Normal Distribution).
        max_beta
    }
}

/// cyclic variable beta generator for traing VAE
pub struct BetaCyclic {
    num_zeros: usize, // num epochs with beta=0
    speed: f32,       // speed of sin cycle
    beta_max: f32,    // maximum beta value
    step: usize,
}

impl BetaCyclic {
    pub fn new(num_zeros: usize, speed: f32, beta_max: f32) -> Self {
        Self {
            num_zeros,
            speed,
            beta_max,
            step: 0,
        }
    }
}

impl Iterator for BetaCyclic {
    type Item = f32;

    fn next(&mut self) -> std::option::Option<f32> {
        let value = if self.step < self.num_zeros {
            0.0
        } else {
            let t = (self.step - self.num_zeros) as f32;
            (t * self.speed).sin() * self.beta_max
        };

        self.step += 1;

        // restart when sin reaches full π/2 cycle
        let cycle_len = self.num_zeros + ((core::f32::consts::PI) / self.speed / 2.0) as usize;
        if self.step >= cycle_len {
            self.step = 0; // repeat cycle
        }
        Some(value)
    }
}
