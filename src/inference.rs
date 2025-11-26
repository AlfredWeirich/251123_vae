#![recursion_limit = "256"]

// --- IMPORTS ---
#[cfg(feature = "conv")]
use vae::conv_model::{ConvVAE, ConvVaeConfig};
#[cfg(feature = "dense")]
use vae::dense_model::{DenseVAE, DenseVaeConfig};
use vae::{LATENT_DIM, MNIST_DIM_X, MNIST_DIM_Y};

use burn::{
    config::Config,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Distribution, Tensor, TensorData},
};
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::Parser;
use image::{GrayImage, Luma};
use std::path::Path;

// --- CLI DEFINITION ---

/// Command-line interface for the VAE inference tool.
///
/// Allows the user to:
/// - Provide a custom latent mean vector (`--mu`)
/// - Optionally provide a custom log-variance vector (`--sigma`)
/// - Specify an output file path
///
/// If `--mu` is omitted, a standard normal latent vector is sampled.
/// If `--mu` is provided without `--sigma`, the generation becomes deterministic
/// (i.e., no reparameterization noise).
#[derive(Parser, Debug)]
#[command(author, version, about = "VAE Inference Generator")]
struct Args {
    /// List of floats corresponding to the mean (μ) of the latent distribution.
    ///
    /// Example:
    /// `--mu 0.5 -1.2 0.3 ...`
    ///
    /// Must contain exactly `LATENT_DIM` values.
    #[arg(long, value_delimiter = ' ', num_args = LATENT_DIM, allow_hyphen_values = true)]
    mu: Option<Vec<f32>>,

    /// List of floats representing log-variance entries.
    ///
    /// Used only if `--mu` is also provided. When supplied, the reparameterization
    /// trick is applied:
    /// ```text
    /// z = mu + eps * exp(0.5 * logvar)
    /// ```
    ///
    /// Length must match `latent_dim`.
    #[arg(long, value_delimiter = ' ', num_args = 1.., allow_hyphen_values = true)]
    sigma: Option<Vec<f32>>,

    /// File path for the saved output image.
    ///
    /// Defaults to `"generated_digit.png"`.
    #[arg(short, long, default_value = "generated_digit.png")]
    output: String,
}

// --- HELPER FUNCTIONS ---

/// Saves a flattened vector of floating-point pixel values as a PNG image.
///
/// Expects:
/// - `pixels`: A `Vec<f32>` of length MNIST_DIM_X × MNIST_DIM_Y
/// - Values in the range `[0.0, 1.0]`
/// - Grayscale output using the Luma channel
///
/// # Errors
///
/// Panics if image writing fails.
fn save_image_as_png(pixels: Vec<f32>, filename: &str) {
    let mut img = GrayImage::new(MNIST_DIM_X, MNIST_DIM_Y);

    for y in 0..MNIST_DIM_X {
        for x in 0..MNIST_DIM_Y {
            // Convert (x, y) → flat index
            let index = (y * MNIST_DIM_X + x) as usize;
            if index >= pixels.len() {
                break;
            }

            // Scale [0,1] → [0,255] and clamp
            let pixel_val = pixels[index];
            let pixel_u8 = (pixel_val * 255.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Luma([pixel_u8]));
        }
    }

    img.save(Path::new(filename)).expect("Failed to save image");
    println!("Saved image to {}", filename);
}

// --- MAIN EXECUTION ---

/// Entry point for the VAE inference program.
///
/// Steps:
/// 1. Parse command-line arguments
/// 2. Load VAE configuration and pretrained weights
/// 3. Build latent vector (`z`) from user input or random sampling
/// 4. Decode `z` into an image using the model's decoder
/// 5. Convert the output tensor into host memory
/// 6. Write image to disk as PNG
fn main() {
    // 0. Parse Command Line Arguments
    let args = Args::parse();

    // 1. Setup Compute Backend
    // WGPU offers GPU acceleration if available.
    type InferenceBackend = Wgpu;
    let device = WgpuDevice::DefaultDevice;

    println!("Loading config...");

    // Load model configuration according to the enabled feature
    #[cfg(feature = "dense")]
    let config = DenseVaeConfig::load("vae_model/mnist_vae.config.json")
        .expect("Config file not found. Did you train the model?");
    #[cfg(feature = "conv")]
    let config = ConvVaeConfig::load("vae_model/mnist_vae.config.json")
        .expect("Config file not found. Did you train the model?");

    // 2. Load Pre-trained Model Weights
    // Ensures full precision for deterministic reconstruction.
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load("vae_model/mnist_vae".into(), &device)
        .expect("Model weights file not found");

    // Initialize model instance and load weights
    #[cfg(feature = "dense")]
    let model = DenseVAE::<InferenceBackend>::new(&config, &device).load_record(record);
    #[cfg(feature = "conv")]
    let model = ConvVAE::<InferenceBackend>::new(&config, &device).load_record(record);

    println!("Model loaded successfully.");

    // 3. Determine Latent Vector (z)
    //
    // Two modes:
    // - MANUAL INPUT: user supplies --mu [and optionally --sigma]
    // - RANDOM MODE: z ~ N(0, 1)
    let z = if let Some(mu_vec) = args.mu {
        // --- MANUAL INPUT MODE ---

        if mu_vec.len() != config.latent_dim {
            panic!(
                "Error: Provided --mu has {} elements, but model requires latent_dim={}",
                mu_vec.len(),
                config.latent_dim
            );
        }

        println!("Using provided mu vector.");
        let mu = Tensor::<InferenceBackend, 2>::from_floats(
            TensorData::new(mu_vec, vec![1, config.latent_dim]),
            &device,
        );

        if let Some(lv_vec) = args.sigma {
            // Reparameterization only occurs if sigma is supplied.
            if lv_vec.len() != config.latent_dim {
                panic!("Error: --log-var length mismatch.");
            }
            println!("Applying reparameterization with provided sigma.");

            let sigma = Tensor::<InferenceBackend, 2>::from_floats(
                TensorData::new(lv_vec, vec![1, config.latent_dim]),
                &device,
            );

            // Compute: z = mu + eps * exp(0.5 * sigma)
            let std = sigma.mul_scalar(0.5).exp();
            let eps = Tensor::random_like(&std, Distribution::Normal(0.0, 1.0));
            mu + eps * std
        } else {
            // Deterministic latent vector (center of distribution)
            println!("No sigma provided. Generating deterministic image (z = mu).");
            mu
        }
    } else {
        // --- RANDOM MODE ---
        println!("No inputs provided. Generating random sample from Standard Normal.");
        Tensor::<InferenceBackend, 2>::random(
            [1, config.latent_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        )
    };

    // 4. Decode Latent Vector
    // Converts latent representation `z` back into image space.
    let reconstruction = model.decoder.forward(z);

    // 5. Convert Tensor to CPU Vector
    let data = reconstruction.into_data();
    let pixel_data: Vec<f32> = data
        .to_vec()
        .expect("Failed to convert tensor data to vector");

    // 6. Save Output
    save_image_as_png(pixel_data, &args.output);
}
