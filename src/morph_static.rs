#![recursion_limit = "256"]

// --- IMPORTS ---
#[cfg(feature = "conv")]
use vae::conv_model::{ConvVAE, ConvVaeConfig};
#[cfg(feature = "dense")]
use vae::dense_model::{DenseVAE, DenseVaeConfig};
use vae::{MNIST_DIM_X, MNIST_DIM_Y, build_rgb_bytes};

use burn::{
    config::Config,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, TensorData},
};
use burn_wgpu::{Wgpu, WgpuDevice};

// Added clap for CLI parsing
use clap::Parser;
use tiny_plot_lib::{GridItem, MultiChart, RawImage};

// --- CLI DEFINITION ---

/// Command-line arguments for the VAE grid generator.
///
/// This tool generates a `grid_rows × grid_cols` visualization of decoded
/// latent vectors, with the first two latent dimensions varying across grid
/// coordinates and the third latent dimension controlled via CLI.
///
/// Example:
/// ```bash
/// cargo run --release -- --z2 1.5
/// ```
#[derive(Parser, Debug)]
#[command(author, version, about = "VAE Grid Generator")]
struct Args {
    /// The value used for the 3rd latent dimension (z[2]).
    ///
    /// This creates a "slice" through latent space, so the grid explores only
    /// dimensions 0 and 1 while dimension 2 stays constant.
    #[arg(long, default_value_t = 2.0, allow_hyphen_values = true)]
    z2: f32,
}

/// Entry point for the static grid VAE visualization.
///
/// This program:
/// 1. Loads a pretrained VAE model (Dense or Conv depending on feature flag).
/// 2. Builds a 2D grid of latent vectors spanning a specified range.
/// 3. Sets the 3rd latent dimension to a user-specified CLI value.
/// 4. Decodes all latent vectors individually.
/// 5. Displays the full image grid using `tiny_plot_lib`.
fn main() {
    // 0. Parse CLI Arguments
    let args = Args::parse();

    // 1. Setup Compute Backend (WGPU for GPU acceleration)
    type InferenceBackend = Wgpu;
    let device = WgpuDevice::DefaultDevice;

    println!("Loading config...");

    // Load model configuration
    #[cfg(feature = "dense")]
    let config = DenseVaeConfig::load("vae_model/mnist_vae.config.json")
        .expect("Config file not found. Did you train the model?");
    #[cfg(feature = "conv")]
    let config = ConvVaeConfig::load("vae_model/mnist_vae.config.json")
        .expect("Config file not found. Did you train the model?");

    // 2. Load Pre-trained Model Weights
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load("vae_model/mnist_vae".into(), &device)
        .expect("Model weights file not found");

    // Instantiate model type depending on compile-time feature
    #[cfg(feature = "dense")]
    let model = DenseVAE::<InferenceBackend>::new(&config, &device).load_record(record);
    #[cfg(feature = "conv")]
    let model = ConvVAE::<InferenceBackend>::new(&config, &device).load_record(record);

    println!("Model loaded successfully.");
    println!("Generating 15x15 grid with z[2] = {}", args.z2);

    // 3. Grid Settings
    //
    // Latent grid shape: 15×15
    // Latent coords mapped linearly from [-3, +3]
    let grid_rows = 15;
    let grid_cols = 15;
    let range_min = -3.0f32;
    let range_max = 3.0f32;

    let mut items: Vec<GridItem> = Vec::with_capacity(grid_rows * grid_cols);

    // 4. Iterate over the grid
    //
    // For each grid cell, we:
    // - Compute latent coords (z_x, z_y) using linear interpolation
    // - Build latent vector [z_x, z_y, z2, 0, ...]
    // - Decode image from VAE
    // - Convert to RGB for plotting library
    for y in 0..grid_rows {
        for x in 0..grid_cols {
            // Linear interpolation (0.0 → 1.0 across axis)
            let fx = x as f32 / (grid_cols - 1) as f32;
            let fy = y as f32 / (grid_rows - 1) as f32;

            // Map interpolation into [-3, +3]
            let z_x = range_min + fx * (range_max - range_min);
            let z_y = range_min + fy * (range_max - range_min);

            // Initialize latent vector
            let mut z_vec = vec![0.0; config.latent_dim];

            // Assign X/Y dims
            if config.latent_dim >= 2 {
                z_vec[0] = z_x;
                z_vec[1] = z_y;
            } else {
                // Fallback: if latent_dim == 1, use first dim only
                z_vec[0] = z_x;
            }

            // Set 3rd dimension from CLI argument
            if config.latent_dim > 2 {
                z_vec[2] = args.z2;
            }

            // Convert into a 1×latent_dim tensor
            let z_tensor = Tensor::<InferenceBackend, 2>::from_floats(
                TensorData::new(z_vec, vec![1, config.latent_dim]),
                &device,
            );

            // Inference: decode latent → image tensor
            let reconstruction = model.decoder.forward(z_tensor);
            let data = reconstruction.into_data();

            // Extract raw pixel data (grayscale floats)
            let pixel_data: Vec<f32> = data.to_vec().expect("Tensor data error");

            // Push as an RGB RawImage into the grid
            items.push(
                RawImage {
                    title: "".into(),
                    width: MNIST_DIM_X,
                    height: MNIST_DIM_Y,
                    pixels: build_rgb_bytes(&pixel_data),
                }
                .into(),
            );
        }
    }

    // 5. Display Visualization Window
    //
    // `tiny_plot_lib::run_static` requires &'static str, so we leak a string.
    let title = format!("VAE Grid (z[2]={})", args.z2);
    let leaked_title: &'static str = Box::leak(title.into_boxed_str());

    let m_chart = MultiChart::new(items);

    println!("Opening Window...");
    let _ = tiny_plot_lib::run_static(leaked_title, m_chart);
}

