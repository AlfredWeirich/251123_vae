#![recursion_limit = "256"]

// --- IMPORTS ---
use iced::{Element, Length, Subscription, Task, time, widget::container};
use std::time::Instant;

// Burn Imports
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

// Plot Lib Imports (Your local lib)
use tiny_plot_lib::{GridItem, MultiChart, RawImage};

// --- CONSTANTS ---

/// Number of rows in the visualization grid.
const GRID_ROWS: usize = 10;

/// Number of columns in the visualization grid.
const GRID_COLS: usize = 10;

/// Total number of images displayed per frame (rows × cols).
const GRID_SIZE: usize = GRID_ROWS * GRID_COLS;

/// Latent range span for grid axes (each dimension varies in [-LATENT_RANGE, LATENT_RANGE]).
const LATENT_RANGE: f32 = 3.0;

// --- MAIN ---

/// Entry point for the interactive VAE latent space explorer.
///
/// This application:
/// 1. Loads a pretrained VAE model.
/// 2. Creates a `GRID_ROWS × GRID_COLS` grid of sampled latent vectors.
/// 3. Continuously oscillates the 3rd latent dimension (if present) using a sinusoid.
/// 4. Runs inference on all latent vectors as a batch every 100ms.
/// 5. Displays the decoded images in real-time using an Iced UI.
///
/// Feature flags determine which model type is compiled (`dense` or `conv`).
fn main() -> iced::Result {
    // 1. Setup Backend (WGPU for hardware acceleration)
    type InferenceBackend = Wgpu;
    let device = WgpuDevice::DefaultDevice;

    println!("Loading model...");

    // 2. Load VAE config and pretrained weights (before starting UI)
    #[cfg(feature = "dense")]
    let config = DenseVaeConfig::load("vae_model/mnist_vae.config.json").expect("Config not found");
    #[cfg(feature = "conv")]
    let config = ConvVaeConfig::load("vae_model/mnist_vae.config.json").expect("Config not found");

    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load("vae_model/mnist_vae".into(), &device)
        .expect("Weights not found");

    // Instantiate model and load weights
    #[cfg(feature = "dense")]
    let model = DenseVAE::<InferenceBackend>::new(&config, &device).load_record(record);
    #[cfg(feature = "conv")]
    let model = ConvVAE::<InferenceBackend>::new(&config, &device).load_record(record);

    // 3. Initialize the grid UI with placeholder black images
    let mut items = Vec::with_capacity(GRID_SIZE);
    for _ in 0..GRID_SIZE {
        items.push(GridItem::Image(RawImage {
            title: "".into(),
            width: MNIST_DIM_X,
            height: MNIST_DIM_Y,
            pixels: vec![0; (MNIST_DIM_X * MNIST_DIM_Y * 3) as usize], // RGB: black
        }));
    }

    // Assemble Application State
    let app = VaeApp {
        model,
        device,
        multi_chart: MultiChart::new(items),
        start_time: Instant::now(),
        latent_dim: config.latent_dim,
        current_z2: 0.0,
    };

    println!("Starting UI...");

    // Start the Iced application
    iced::application("Dynamic VAE Latent Space", VaeApp::update, VaeApp::view)
        .subscription(VaeApp::subscription)
        .antialiasing(true)
        .run_with(|| (app, Task::none()))
}

// --- APP STATE ---

/// Type alias used to switch between dense/conv VAE models depending on compile-time features.
#[cfg(feature = "dense")]
type ModelType = DenseVAE<Wgpu>;
#[cfg(feature = "conv")]
type ModelType = ConvVAE<Wgpu>;

/// Main state for the VAE visualization UI.
///
/// Holds:
/// - The VAE model
/// - Device information
/// - A grid-based chart container
/// - Oscillation state & timing
/// - Dynamic latent dimension controls
struct VaeApp {
    model: ModelType,
    device: WgpuDevice,
    multi_chart: MultiChart,
    start_time: Instant,
    latent_dim: usize,
    current_z2: f32,
}

/// UI message type used for timer ticks.
#[derive(Debug, Clone, Copy)]
enum Message {
    /// Fired every ~100ms via Iced subscription.
    Tick(Instant),
}

impl VaeApp {
    /// Updates the UI/application state in response to messages.
    ///
    /// The `Tick` message drives the animation:
    /// - Computes a new sinusoidal z₂ value
    /// - Constructs a full batch of latent vectors (GRID_SIZE × latent_dim)
    /// - Runs the decoder on this batch
    /// - Updates the on-screen grid with newly generated images
    fn update(&mut self, message: Message) {
        match message {
            Message::Tick(now) => {
                // 1. Compute dynamic oscillating Z₂ via a sine curve
                let elapsed = (now - self.start_time).as_secs_f32();
                self.current_z2 = (elapsed * 0.5).sin() * LATENT_RANGE;

                // 2. Build the latent batch
                //
                //    Batch size = GRID_SIZE
                //    Each entry = [latent_dim] vector
                //
                //    Dimensions 0 and 1 come from grid coordinates,
                //    Dimension 2 oscillates over time,
                //    Remaining dimensions = 0.
                let mut batch_vec = Vec::with_capacity(GRID_SIZE * self.latent_dim);

                for y in 0..GRID_ROWS {
                    for x in 0..GRID_COLS {
                        let fx = x as f32 / (GRID_COLS - 1) as f32;
                        let fy = y as f32 / (GRID_ROWS - 1) as f32;

                        let z_x = -LATENT_RANGE + fx * (LATENT_RANGE * 2.0);
                        let z_y = -LATENT_RANGE + fy * (LATENT_RANGE * 2.0);

                        // Fill dimensions carefully up to latent_dim
                        batch_vec.push(z_x); // dim 0
                        if self.latent_dim > 1 {
                            batch_vec.push(z_y);
                        } // dim 1
                        if self.latent_dim > 2 {
                            batch_vec.push(self.current_z2);
                        } // dim 2

                        // Fill extra dims with zero
                        for _ in 3..self.latent_dim {
                            batch_vec.push(0.0);
                        }
                    }
                }

                // 3. Convert latent batch → a single tensor
                let z_tensor = Tensor::<Wgpu, 2>::from_floats(
                    TensorData::new(batch_vec, vec![GRID_SIZE, self.latent_dim]),
                    &self.device,
                );

                // Decode via the model's decoder
                let reconstruction = self.model.decoder.forward(z_tensor);

                // 4. Extract tensor data into CPU memory
                //
                // Output shape: [GRID_SIZE, 1, 28, 28]
                let data_stream = reconstruction.into_data();
                let flat_pixels: Vec<f32> = data_stream.to_vec().unwrap();

                // 5. Update individual grid images
                let img_size = (MNIST_DIM_X * MNIST_DIM_Y) as usize;

                for i in 0..GRID_SIZE {
                    let start = i * img_size;
                    let end = start + img_size;
                    if end <= flat_pixels.len() {
                        let img_slice = &flat_pixels[start..end];

                        // Convert grayscale float → RGB byte triplets
                        let rgb_bytes = build_rgb_bytes(img_slice);

                        // Push update to the chart library
                        self.multi_chart.update_image_pixels(i, rgb_bytes);
                    }
                }
            }
        }
    }

    /// Renders the UI.  
    /// Displays the multi-chart grid that visualizes the decoded images.
    fn view(&self) -> Element<'_, Message> {
        let content = self.multi_chart.to_element();

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(10)
            .into()
    }

    /// Generates a timer subscription firing ~every 100ms.
    ///
    /// This drives the smooth real-time latent space exploration animation.
    fn subscription(&self) -> Subscription<Message> {
        time::every(std::time::Duration::from_millis(100)).map(Message::Tick)
    }
}
