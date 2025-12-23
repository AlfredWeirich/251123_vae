#![recursion_limit = "256"]

//! # MNIST Dataset Viewer
//!
//! This utility application allows for the visual inspection of the MNIST dataset
//! (Modified National Institute of Standards and Technology database).
//!
//! It fetches a batch of test images, processes them, and displays them in a grid.
//! A key feature of this viewer is **Semantic Colorization**: instead of standard grayscale,
//! images are tinted with a specific color corresponding to their ground-truth label (0-9).
//! This makes it easy to visually verify label accuracy and distinguish classes.

use burn::data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset};
use burn::tensor::backend::Backend;
use vae::mnist_data::MnistBatcher;
use vae::{MNIST_DIM_X, MNIST_DIM_Y};
/// Total number of pixels per image (28 * 28 = 784).
const IMAGE_SIZE: usize = (MNIST_DIM_X * MNIST_DIM_Y) as usize;

// --- CONSTANTS ---

/// Number of rows in the visualization grid.
/// Number of columns in the visualization grid.
const GRID_ROWS: usize = 10;
const GRID_COLS: usize = 10;

/// Upscaling factor for visualization.
///
/// MNIST images are very small (28x28). We upscale them by this factor (e.g., 3x)
/// using proper interpolation to make them clearer in the UI.
const RESIZE_FACTOR: u32 = 3;

/// Semantic Color Palette.
///
/// Maps digits `0-9` to distinct RGB color factors `(R, G, B)` where values range `[0.0, 1.0]`.
/// Used to tint grayscale images based on their classification label.
const LABEL_PALETTE: [(f32, f32, f32); 10] = [
    (1.0, 0.0, 0.0), // 0: Red
    (1.0, 0.5, 0.0), // 1: Orange
    (1.0, 1.0, 0.0), // 2: Yellow
    (0.0, 1.0, 0.0), // 3: Green
    (0.0, 1.0, 1.0), // 4: Cyan
    (0.0, 0.0, 1.0), // 5: Blue
    (0.5, 0.0, 1.0), // 6: Purple
    (1.0, 0.0, 1.0), // 7: Magenta
    (0.5, 0.2, 0.0), // 8: Brown
    (1.0, 1.0, 1.0), // 9: White
];

/// Loads a batch of MNIST data and launches the visualization UI.
///
/// # Workflow
/// 1. Initializes the Burn Data Loader for the MNIST Test set.
/// 2. Fetches a single batch containing `GRID_ROWS * GRID_COLS` images.
/// 3. Normalizes pixel values from `[0.0, 1.0]` back to `[0.0, 255.0]`.
/// 4. Zips image data with label data.
/// 5. Transformation: Colorizes and resizes each image.
/// 6. Launches the `tiny_plot_lib` static runner.
///
/// # Type Parameters
/// * `B`: The Burn Backend (e.g., Wgpu, NdArray) used to load the tensors.
pub fn view_first_mnists<B: Backend>(_device: &B::Device) {
    // Initialize the batcher (handles tensor creation)
    let test_batcher = MnistBatcher::<B>::new();

    // Configure the Data Loader
    // We explicitly set the batch size to exactly fill our grid.
    let test_loader = DataLoaderBuilder::new(test_batcher)
        .batch_size(GRID_COLS * GRID_ROWS)
        .build(MnistDataset::test());

    println!("Starting viewing pipeline...");

    // Fetch the first batch from the iterator.
    // Returns (Tensor<Images>, Tensor<Labels>)
    let (train_images, train_labels) = test_loader.iter().next().expect("Dataset is empty");

    // Rescale floating point tensors [0, 1] -> [0, 255] for standard image processing
    let train_images = train_images * 255.0;

    println!("Batch Shape: {:?}", train_images.shape());

    // Move data from GPU/Backend to CPU Vectors
    let images: Vec<f32> = train_images
        .to_data()
        .to_vec()
        .expect("Get train image data");
    let labels: Vec<i32> = train_labels
        .to_data()
        .to_vec()
        .expect("Get train label data");

    let mut mnist_digits: Vec<tiny_plot_lib::GridItem> = Vec::with_capacity(GRID_COLS * GRID_ROWS);

    // Process images and labels in parallel
    // We slice the flat `images` vector into chunks of 784 pixels.
    for (mnist_image, &mnist_label) in images.chunks(IMAGE_SIZE).zip(labels.iter()) {
        let img = create_colored_image(mnist_image, mnist_label);
        mnist_digits.push(tiny_plot_lib::GridItem::Image(img));
    }

    // Configure the MultiChart grid
    let digits_matrix = tiny_plot_lib::MultiChart::new(mnist_digits)
        .with_x_space(4) // Add padding between grid cells
        .with_y_space(4);

    // Launch the Iced Application (Blocks until window closes)
    let _ = tiny_plot_lib::run_static("Test MNIST Images (Colored by Label)", digits_matrix);
}

/// Transforms raw grayscale pixel data into a colorized, resized image struct.
///
/// # Operations
/// 1. **Color Mapping**: Applies an RGB tint based on the `label`.
/// 2. **Buffer Construction**: Converts `f32` intensities to `u8` RGB triplets.
/// 3. **Resizing**: Upscales the image using the `image` crate (Triangle filter) for smoother visualization.
///
/// # Arguments
/// * `image_data`: A slice of 784 `f32` values representing the 28x28 image.
/// * `label`: The classification digit (0-9).
///
/// # Returns
/// A `RawImage` struct compatible with `tiny_plot_lib`.
fn create_colored_image(image_data: &[f32], label: i32) -> tiny_plot_lib::RawImage {
    let width = MNIST_DIM_X;
    let height = MNIST_DIM_Y;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);

    // 1. Retrieve color factors for the specific digit
    // Use modulo 10 to ensure safety, though labels are expected to be 0-9.
    let (r_factor, g_factor, b_factor) = LABEL_PALETTE[label as usize % 10];

    // 2. Build the RGB buffer pixel by pixel
    for &pixel_val in image_data {
        // `pixel_val` ranges 0.0 to 255.0 (grayscale intensity).
        // Multiply intensity by the color factor to tint the digit.
        let r = (pixel_val * r_factor) as u8;
        let g = (pixel_val * g_factor) as u8;
        let b = (pixel_val * b_factor) as u8;

        pixels.extend_from_slice(&[r, g, b]);
    }

    // 3. Create an image buffer wrapper for resizing operations
    let img = image::RgbImage::from_raw(width, height, pixels)
        .expect("Failed to create image buffer from raw pixels");

    // 4. Upscale the image
    // Triangle filtering provides a good balance between speed and smoothness for upscaling.
    let resized = image::imageops::resize(
        &img,
        MNIST_DIM_X * RESIZE_FACTOR,
        MNIST_DIM_Y * RESIZE_FACTOR,
        image::imageops::FilterType::Triangle,
    );

    // Convert back to raw bytes for the plotting library
    let resized_buffer = resized.to_vec();

    tiny_plot_lib::RawImage {
        title: "".into(), // Title left empty for cleaner grid view
        width: width * RESIZE_FACTOR,
        height: height * RESIZE_FACTOR,
        pixels: resized_buffer,
    }
}

/// Entry point for the application.
fn main() {
    // Select the WGPU backend (Metal, Vulkan, DX12, etc.)
    // WGPU is generally preferred for hardware acceleration.
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    // Define the backend type alias
    type MyBackend = Wgpu;

    // Automatically select the best available graphics device
    let device = WgpuDevice::DefaultDevice;

    // Execute the viewer logic
    view_first_mnists::<MyBackend>(&device);
}
