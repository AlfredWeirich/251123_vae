#![recursion_limit = "256"]

use burn::data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset};
use burn::tensor::backend::Backend; // Simplified import
use vae::mnist_data::MnistBatcher;

// --- CONSTANTS ---
const GRID_ROWS: usize = 10;
const GRID_COLS: usize = 10;
const IMAGE_WIDTH: u32 = 28;
const IMAGE_HEIGHT: u32 = 28;
const IMAGE_SIZE: usize = (IMAGE_WIDTH * IMAGE_HEIGHT) as usize;
const RESIZE_FACTOR: u32 = 3;

// Color Palette for digits 0-9 (R, G, B factors from 0.0 to 1.0)
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

fn view_first_mnists<B: Backend>(_device: &B::Device) {
    let test_batcher = MnistBatcher::<B>::new();
    let test_loader = DataLoaderBuilder::new(test_batcher)
        .batch_size(GRID_COLS * GRID_ROWS)
        .build(MnistDataset::test());

    println!("Starting viewing...");

    // Get the first batch
    let (train_images, train_labels) = test_loader.iter().next().unwrap();

    // Scale 0..1 to 0..255
    let train_images = train_images * 255.0;

    println!("Batch Shape: {:?}", train_images.shape());

    // Convert tensors to standard vectors
    let images: Vec<f32> = train_images.to_data().to_vec().expect("get images");
    let labels: Vec<i32> = train_labels.to_data().to_vec().expect("get labels");

    let mut items: Vec<tiny_plot_lib::GridItem> = Vec::with_capacity(GRID_COLS * GRID_ROWS);

    // ZIP images with labels to process them together
    for (image_chunk, &label) in images.chunks(IMAGE_SIZE).zip(labels.iter()) {
        let img = create_colored_image(image_chunk, label);
        items.push(tiny_plot_lib::GridItem::Image(img));
    }

    let chart = tiny_plot_lib::MultiChart::new(items);

    // Run the UI
    let _ = tiny_plot_lib::run_static("Test MNIST Images (Colored by Label)", chart);
}

fn create_colored_image(image_data: &[f32], label: i32) -> tiny_plot_lib::RawImage {
    let width = IMAGE_WIDTH;
    let height = IMAGE_HEIGHT;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);

    // 1. Get color factors for this label (modulo 10 just to be safe)
    let (r_factor, g_factor, b_factor) = LABEL_PALETTE[label as usize % 10];

    for &pixel_val in image_data {
        // pixel_val is 0.0 to 255.0 (grayscale intensity)
        // We multiply the intensity by the color factor.

        let r = (pixel_val * r_factor) as u8;
        let g = (pixel_val * g_factor) as u8;
        let b = (pixel_val * b_factor) as u8;

        pixels.extend_from_slice(&[r, g, b]);
    }
    let img = image::RgbImage::from_raw(width, height, pixels).unwrap();
    let resized = image::imageops::resize(
        &img,
        IMAGE_WIDTH * RESIZE_FACTOR,
        IMAGE_HEIGHT * RESIZE_FACTOR,
        image::imageops::FilterType::Triangle,
    );
    let resized_buffer = resized.to_vec();

    // Assuming you are using the struct init style from your snippet
    tiny_plot_lib::RawImage {
        title: "".into(), // Optional: Put the label in the title
        width: width * RESIZE_FACTOR,
        height: height * RESIZE_FACTOR,
        pixels: resized_buffer,
    }

    // OR if you updated the lib to use the constructor:
    // tiny_plot_lib::RawImage::new(format!("{}", label), width, height, pixels)
}

fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    // 1. Define Backend
    type MyBackend = Wgpu;

    // 2. Select Device
    let device = WgpuDevice::DefaultDevice;

    // 3. Run
    view_first_mnists::<MyBackend>(&device);
}
