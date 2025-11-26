#![recursion_limit = "256"]

use burn::data::dataloader::DataLoader;
use burn::prelude::ElementConversion;
use burn::prelude::ToElement;
use burn::record::FullPrecisionSettings;
use burn::tensor::{Tensor, backend::Backend};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::BinFileRecorder, // Import for saving/loading
    tensor::backend::AutodiffBackend,
};

use vae::LATENT_DIM;
use vae::ModelTreePrinter;
#[cfg(feature = "conv")]
use vae::conv_model::{ConvVAE, ConvVaeConfig};
#[cfg(feature = "dense")]
use vae::dense_model::{DenseVAE, DenseVaeConfig};
use vae::loss_function;
use vae::mnist_data::MnistBatcher;

// Use WGPU backend instead of NdArray
use burn_wgpu::{Wgpu, WgpuDevice};

use std::sync::Arc;

// --- Training Loop ---

/// Executes the main training loop for the Variational Autoencoder.
///
/// This function handles the following lifecycle:
/// 1. **Configuration:** Sets up model hyperparameters based on feature flags (Dense vs Conv).
/// 2. **Initialization:** Creates the model, optimizer (Adam), and data loaders.
/// 3. **Epoch Loop:** Iterates through the dataset, computing the VAE loss (Reconstruction + KL Divergence).
/// 4. **Optimization:** Performs backpropagation and updates model weights.
/// 5. **Persistence:** Saves the trained model weights and configuration to disk.
///
/// # Type Parameters
/// * `B`: The AutodiffBackend to use (e.g., WGPU with Autodiff), allowing for gradient computation.
///
/// # Arguments
/// * `device`: The specific compute device (CPU/GPU) where tensors will be allocated.
fn training_loop<B: AutodiffBackend>(device: B::Device) {
    // Initialize configuration based on compile-time feature flags.
    // This allows switching between a fully connected (Dense) or Convolutional architecture.
    #[cfg(feature = "dense")]
    let config = DenseVaeConfig::new().with_latent_dim(LATENT_DIM);
    #[cfg(feature = "conv")]
    let config = ConvVaeConfig::new().with_latent_dim(LATENT_DIM);

    // Override default latent dimension with global constant
    // config.latent_dim = LATENT_DIM;
    println!("Configuration: {:?}", config);
    println!("Using Device: {:?}", device);

    // Instantiate the VAE model structure on the specified device
    #[cfg(feature = "dense")]
    let mut model = DenseVAE::<B>::new(&config, &device);
    #[cfg(feature = "conv")]
    let mut model = ConvVAE::<B>::new(&config, &device);

    let mut visitor = ModelTreePrinter::new();

    // This triggers the traversal
    model.visit(&mut visitor);

    // Configure the Adam optimizer with Weight Decay (L2 Regularization)
    // to prevent overfitting and stabilize training.
    let optim_config =
        AdamConfig::new().with_weight_decay(Some(burn::optim::decay::WeightDecayConfig {
            penalty: 1e-5,
        }));
    let mut optimizer = optim_config.init();

    // Prepare the Data Loader
    // MnistBatcher handles converting raw images into tensors usable by the backend.
    let batcher = MnistBatcher::<B>::new();
    let dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(42) // Fixed seed for reproducibility during shuffling
        .num_workers(8) // Parallel data loading
        .build(MnistDataset::train());
    // Setup Test Loader (Validation)
    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .build(MnistDataset::test());

    println!("Starting training...");

    for epoch in 1..=config.num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        //let mut batch_num = 0;
        //for batch in dataloader.iter() {

        // Iterate over batches. Each 'batch' is a tensor of flattened images.
        for (batch_num, batch) in dataloader.iter().enumerate() {
            let x_flat = batch;

            #[cfg(feature = "conv")]
            // Reshape for Conv Network: [B, 784] -> [B, 1, 28, 28]
            // Convolutional layers require (Batch, Channels, Height, Width) format.
            let x_img = x_flat.clone().reshape([0, 1, 28, 28]);

            // Forward pass: Returns reconstructed input, mean (mu), and log-variance (sigma)
            #[cfg(feature = "dense")]
            let (recon_x, mu, sigma) = model.forward(x_flat.clone());
            #[cfg(feature = "conv")]
            let (recon_x, mu, sigma) = model.forward(x_img.clone());

            // Debug print: Inspect latent variables for the first batch of the epoch.
            // Helpful to ensure values aren't exploding (NaN) or collapsing to zero.
            if batch_num == 0 {
                println!(
                    "----------------------------mu: {:?}",
                    mu.to_data()
                        .as_slice::<f32>()
                        .expect("GET MU")
                        .get(0..LATENT_DIM)
                );
                println!(
                    "----------------------------log: {:?}",
                    sigma
                        .to_data()
                        .as_slice::<f32>()
                        .expect("GET MU")
                        .get(0..LATENT_DIM)
                );
            }
            //batch_num += 1;

            // Compute VAE Loss:
            // 1. Reconstruction Loss (MSE or BCE): How close is output to input?
            // 2. KL Divergence: How close is the latent distribution to N(0,1)?
            let loss = loss_function(recon_x, x_flat, mu, sigma);

            // Track metrics
            let loss_val = loss.clone().into_scalar().to_f64();
            total_loss += loss_val;
            batch_count += 1;

            // Backward pass & Optimization step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        // Log average loss per epoch
        let avg_loss = total_loss / batch_count as f64;
        println!(
            "Epoch [{}/{}], Loss: {:.4}",
            epoch, config.num_epochs, avg_loss
        );
        // --- VALIDATION PHASE ---
        // We pass a reference to the model so it isn't consumed
        let val_loss = validate(&model, test_loader.clone());

        println!("Epoch {} | Validation Loss: {:.4}", epoch, val_loss);
    }

    // --- SAVING THE MODEL ---
    println!("Saving model...");
    // We use FullPrecisionSettings for maximum compatibility, though HalfPrecision is an option for size.
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    // This creates "mnist_vae.bin.gz" (compressed binary)
    // Saves the learnable parameters (weights and biases)
    model
        .save_file("vae_model/mnist_vae", &recorder)
        .expect("Failed to save model");

    // Also save config for reference to ensure we can load the model
    // with the exact same architecture structure later.
    config
        .save("vae_model/mnist_vae.config.json")
        .expect("Failed to save config");

    println!("Model saved to mnist_vae.bin.gz");
}

/// Runs validation on the test dataset
pub fn validate<B: Backend>(
    #[cfg(feature = "dense")] model: &DenseVAE<B>, // Or your specific struct name
    #[cfg(feature = "conv")] model: &ConvVAE<B>,   // Or your specific struct name
    dataloader: Arc<dyn DataLoader<B, Tensor<B, 2>>>, // Adjust type to your Batch struct
) -> f64 {
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    // Iterate over test data
    for batch in dataloader.iter() {
        let x_flat = batch; // Shape: [batch_size, 784] (or 4D if conv)

        #[cfg(feature = "conv")]
        // Reshape for Conv Network: [B, 784] -> [B, 1, 28, 28]
        // Convolutional layers require (Batch, Channels, Height, Width) format.
        let x_img = x_flat.clone().reshape([0, 1, 28, 28]);

        // Forward pass: Returns reconstructed input, mean (mu), and log-variance (sigma)
        #[cfg(feature = "dense")]
        let (recon_x, mu, sigma) = model.forward(x_flat.clone());
        #[cfg(feature = "conv")]
        let (recon_x, mu, sigma) = model.forward(x_img.clone());

        // 2. Calculate loss
        // Assuming x is already flattened if your model is Dense,
        // or your model handles flattening (as per previous refactor).
        let loss = loss_function(recon_x, x_flat, mu, sigma);

        // 3. Accumulate scalar value
        // .into_scalar() brings the data back to CPU f64
        total_loss += loss.into_scalar().elem::<f64>();
        num_batches += 1;
    }

    // Return average loss
    total_loss / (num_batches as f64)
}

fn main() {
    // Change Backend to Wgpu (Metal on Mac, Vulkan/DX12 on Windows/Linux)
    // Autodiff wrapper is required for training to enable gradient tracking.
    type TrainBackend = burn::backend::Autodiff<Wgpu>;

    // Select Device
    // 'DefaultDevice' usually picks the most powerful GPU available (e.g., M-series on Mac)
    let device = WgpuDevice::DefaultDevice;

    training_loop::<TrainBackend>(device);
}

// type MyBackend = burn::backend::Autodiff<NdArray>;
// let device = NdArrayDevice::Cpu;
