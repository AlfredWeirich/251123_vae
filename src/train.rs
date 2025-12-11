#![recursion_limit = "256"]

//! # MNIST Variational Autoencoder (VAE) Training Script
//!
//! This crate implements the training pipeline for a VAE using the [Burn](https://burn.dev) framework.
//! It supports dynamic architecture switching (Dense vs. Convolutional) via feature flags and provides
//! advanced hyperparameter scheduling for the KL-Divergence weight ($\beta$).
//!
//! ## Key Features
//! - **Dual Architecture**: Switch between MLP (`feature = "dense"`) and CNN (`feature = "conv"`) at compile time.
//! - **Autodiff Support**: Uses `burn-wgpu` for GPU-accelerated automatic differentiation.
//! - **KL Annealing**: configurable strategies (Fixed, Linear Ramp, Cyclic) to mitigate posterior collapse.
//! - **State Persistence**: Saves model weights and configuration to disk for inference.

use burn::prelude::ToElement;
use burn::{
    Tensor,
    config::Config,
    data::{dataloader::DataLoader, dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::Int,
    tensor::backend::{AutodiffBackend, Backend},
};

// Internal crate imports
use vae::{ModelTreePrinter, loss_function, mnist_data::MnistBatcher};

// Architecture-specific imports controlled by cargo features
#[cfg(feature = "conv")]
use vae::conv_model::{ConvVAE as VaeModel, ConvVaeConfig as VaeConfig};
#[cfg(feature = "dense")]
use vae::dense_model::{DenseVAE as VaeModel, DenseVaeConfig as VaeConfig};

// Backend selection
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::{Parser, ValueEnum};
use std::sync::Arc;

// -------------------------------------------------------------------------------------------------
// CLI Arguments & Configuration
// -------------------------------------------------------------------------------------------------

/// Strategies for scheduling the $\beta$ parameter (KL Divergence weight).
///
/// Modulating $\beta$ during training helps balance reconstruction accuracy against latent space regularity.
#[derive(Debug, Clone, ValueEnum)]
enum BetaStrategy {
    /// Keeps $\beta$ constant throughout training.
    Fixed,
    /// Linearly increases $\beta$ from 0.0 to `beta_target` over the total epochs.
    /// Useful for preventing "posterior collapse" early in training.
    Ramp,
    /// Repeats a ramp cycle multiple times.
    /// Helps the model learn to use latent units that might have become inactive.
    Cyclic,
}

/// Command Line Interface arguments for configuring the VAE training process.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of epochs to train the model.
    #[arg(long, default_value_t = 40)]
    num_epochs: usize,

    /// Dimensionality of the latent space (Z).
    /// Smaller values force higher compression; larger values allow more detail but may reduce generative quality.
    #[arg(long, default_value_t = 3)]
    latent_dim: usize,

    // --- Beta (KL Divergence) Configuration ---
    /// The maximum or target $\beta$ value.
    /// For standard VAEs, this is typically 1.0. For $\beta$-VAEs, this can be >1.0.
    #[arg(long, default_value_t = 0.5)]
    beta_target: f32,

    /// The scheduling strategy for $\beta$.
    #[arg(long, value_enum, default_value_t = BetaStrategy::Fixed)]
    beta_strategy: BetaStrategy,

    /// (Cyclic Only) The number of epochs that constitute one full annealing cycle.
    #[arg(long, default_value_t = 4)]
    cyclic_period: usize,

    /// (Cyclic Only) The fraction of a cycle spent increasing $\beta$ (0.0 to 1.0).
    /// The remainder of the cycle is spent at `beta_target`.
    #[arg(long, default_value_t = 0.5)]
    cyclic_ratio: f32,
}

// -------------------------------------------------------------------------------------------------
// Logic Providers
// -------------------------------------------------------------------------------------------------

/// Helper struct to encapsulate the logic for KL-Divergence annealing.
enum BetaScheduler {
    Fixed(f32),
    Ramp {
        target: f32,
    },
    Cyclic {
        target: f32,
        period: usize,
        ratio: f32,
    },
}

impl BetaScheduler {
    /// Factory method to create the scheduler based on parsed CLI arguments.
    fn from_args(args: &Args) -> Self {
        match args.beta_strategy {
            BetaStrategy::Fixed => BetaScheduler::Fixed(args.beta_target),
            BetaStrategy::Ramp => BetaScheduler::Ramp {
                target: args.beta_target,
            },
            BetaStrategy::Cyclic => BetaScheduler::Cyclic {
                target: args.beta_target,
                period: args.cyclic_period,
                ratio: args.cyclic_ratio,
            },
        }
    }

    /// Calculates the specific $\beta$ value for the current epoch.
    fn step(&self, current_epoch: usize, total_epochs: usize) -> f32 {
        match self {
            BetaScheduler::Fixed(val) => *val,

            BetaScheduler::Ramp { target } => {
                // Linear ramp from 0.0 to target
                let progress = current_epoch as f32 / total_epochs as f32;
                progress * target
            }

            BetaScheduler::Cyclic {
                target,
                period,
                ratio,
            } => {
                // Calculate position within the current cycle (0-indexed based on period)
                let cycle_idx = (current_epoch - 1) % period;
                let cycle_progress = cycle_idx as f32 / *period as f32;

                if cycle_progress < *ratio {
                    // Increasing phase: Linear ramp up
                    (cycle_progress / ratio) * target
                } else {
                    // Plateau phase: Hold steady at target
                    *target
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Training Pipeline
// -------------------------------------------------------------------------------------------------

/// Executes the main training lifecycle for the Variational Autoencoder.
///
/// # Workflow
/// 1. **Setup**: Initializes the model, optimizer, and data loaders.
/// 2. **Loop**: Iterates through epochs and batches.
/// 3. **Forward**: Passes data through the Encoder and Decoder.
/// 4. **Loss**: Calculates Reconstruction Loss + ($\beta$ * KL Divergence).
/// 5. **Backward**: Computes gradients and updates weights using Adam.
/// 6. **Validation**: Evaluates on test data periodically.
/// 7. **Save**: Persists artifacts to disk.
///
/// # Generics
/// * `B`: The `AutodiffBackend` (e.g., WGPU, Torch). It must support automatic differentiation.
fn training_loop<B: AutodiffBackend>(device: B::Device, args: Args) {
    // 1. Configuration
    let config = VaeConfig::new()
        .with_latent_dim(args.latent_dim)
        .with_num_epochs(args.num_epochs);

    println!("Configuration: {:?}", config);
    println!("Using Device: {:?}", device);

    // 2. Model Initialization
    let mut model = VaeModel::<B>::new(&config, &device);

    // Print Model Architecture
    let mut visitor = ModelTreePrinter::new();
    model.visit(&mut visitor);
    println!("Model Params: {}", model.num_params());

    // 3. Data Loading
    let (train_loader, test_loader) = get_dataloader::<B>(config.clone());

    // 4. Optimizer Setup
    // We use L2 Regularization (Weight Decay) to prevent overfitting.
    let optim_config =
        AdamConfig::new().with_weight_decay(Some(burn::optim::decay::WeightDecayConfig {
            penalty: 1e-5,
        }));
    let mut optimizer = optim_config.init();

    // 5. Beta scheduler Setup
    let beta_scheduler = BetaScheduler::from_args(&args);

    println!("Starting training...");

    // Baseline Validation
    let val_loss = validate(&model, test_loader.clone());
    println!("Pre-Train Validation Loss: {:.4}", val_loss);

    // 6. Epoch Loop
    for epoch in 1..=config.num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Update Beta for annealing
        let beta = beta_scheduler.step(epoch, config.num_epochs);

        // Batch Loop
        for (train_images, _labels) in train_loader.iter() {
            // Forward Pass
            let (recon_x, mu, logvar) = forward_pass(&model, train_images.clone());

            // Loss Calculation
            let loss = loss_function(recon_x, train_images, mu, logvar, beta);

            // Logging preparation
            let loss_val = loss.clone().into_scalar().to_f64();
            total_loss += loss_val;
            batch_count += 1;

            // Backward Pass (Backpropagation)
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        // Epoch Logging
        let avg_loss = total_loss / batch_count as f64;
        println!(
            "Epoch [{}/{}], Loss: {:.4}, Beta: {:.3}",
            epoch, config.num_epochs, avg_loss, beta
        );

        // Epoch validation
        let val_loss = validate(&model, test_loader.clone());
        println!("Epoch {} | Validation Loss: {:.4}", epoch, val_loss);
    }

    // 7. Save Model
    save_vae_model(model, config);
}

// -------------------------------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------------------------------

/// Serializes and saves the trained model to `vae_model/`.
///
/// Saves:
/// - `mnist_vae.bin.gz`: Model weights (Full Precision).
/// - `mnist_vae.config.json`: Configuration for reconstruction.
fn save_vae_model<B: Backend>(model: VaeModel<B>, config: VaeConfig) {
    println!("Saving model...");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    model
        .save_file("vae_model/mnist_vae", &recorder)
        .expect("Failed to save model weights");

    config
        .save("vae_model/mnist_vae.config.json")
        .expect("Failed to save model config");

    println!("Model saved successfully to vae_model/");
}

/// Evaluates the model on the test dataset.
///
/// # Mode
/// Runs in **Inference Mode**:
/// - No gradients are tracked (saves memory).
/// - Model is set to `.valid()` (disables Dropout/BatchNorm training behavior).
///
/// # Arguments
/// * `dataloader`: Uses `InnerBackend` to avoid Autodiff overhead during validation.
fn validate<B: Backend + AutodiffBackend>(
    model: &VaeModel<B>,
    dataloader: Arc<
        dyn DataLoader<
                B::InnerBackend,
                (Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 1, Int>),
            >,
    >,
) -> f64 {
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    let model_valid = model.valid();

    for (test_image, _label) in dataloader.iter() {
        let x_flat = test_image;

        // Shape handling for Conv vs Dense
        #[cfg(feature = "conv")]
        let x_input = x_flat.clone().reshape([0, 1, 28, 28]);
        #[cfg(feature = "dense")]
        let x_input = x_flat.clone();

        let (recon_x, mu, sigma) = model_valid.forward(x_input);

        // During validation, we often set beta=0.0 to look purely at reconstruction quality,
        // or we can set it to 1.0. Here we use 0.0 for raw MSE/BCE check.
        let loss = loss_function(recon_x, x_flat, mu, sigma, 0.0);

        total_loss += loss.into_scalar().to_f64();
        num_batches += 1;
    }

    total_loss / (num_batches as f64)
}

/// Helper to handle input reshaping based on the active feature flag.
/// - **Dense**: `[Batch, 784]`
/// - **Conv**: `[Batch, 1, 28, 28]`
fn forward_pass<B: Backend>(
    model: &VaeModel<B>,
    images: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    #[cfg(feature = "conv")]
    let input = images.reshape([0, 1, 28, 28]);

    #[cfg(feature = "dense")]
    let input = images;

    model.forward(input)
}

/// Constructs the DataLoaders for Training and Validation.
///
/// - **Train**: Shuffled, Parallel workers, Autodiff backend.
/// - **Test**: Sequential, InnerBackend (no autodiff).
fn get_dataloader<B: Backend + AutodiffBackend>(
    config: VaeConfig,
) -> (
    Arc<dyn DataLoader<B, (Tensor<B, 2>, Tensor<B, 1, Int>)>>,
    Arc<
        dyn DataLoader<
                B::InnerBackend,
                (Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 1, Int>),
            >,
    >,
) {
    let train_batcher = MnistBatcher::<B>::new();
    let train_loader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(8)
        .build(MnistDataset::train());

    let test_batcher = MnistBatcher::<B::InnerBackend>::new();
    let test_loader = DataLoaderBuilder::new(test_batcher)
        .batch_size(config.batch_size)
        .build(MnistDataset::test());

    (train_loader, test_loader)
}

fn main() {
    // 1. Parse Arguments
    let args = Args::parse();

    // 2. Select Backend (WGPU with Autodiff wrapper)
    type TrainBackend = burn::backend::Autodiff<Wgpu>;
    let device = WgpuDevice::DefaultDevice;

    // 3. Launch
    training_loop::<TrainBackend>(device, args);
}
