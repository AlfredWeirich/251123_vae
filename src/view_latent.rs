#![recursion_limit = "256"]

// --- IMPORTS ---
use burn::data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset};
use burn::tensor::backend::Backend;
use vae::mnist_data::MnistBatcher;

// Conditionally import the model structure based on feature flags.
#[cfg(feature = "conv")]
use vae::conv_model::{ConvVAE, ConvVaeConfig};
#[cfg(feature = "dense")]
use vae::dense_model::{DenseVAE, DenseVaeConfig};

use burn::{
    config::Config,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};

// CLI parsing
use clap::Parser;

// Iced imports for interactivity
use iced::{Element, Event, Length, Point, Subscription, event, mouse, widget::container};

// Tiny Plot Lib imports
use tiny_plot_lib::{
    Chart3D, GridItem, MultiChart, Point3D, PointData, Series, Series3D, SeriesStyle, SingleChart,
    run_static,
};

// --- TYPE ALIASES ---
#[cfg(feature = "dense")]
type VaeModel<B> = DenseVAE<B>;
#[cfg(feature = "dense")]
type VaeConfig = DenseVaeConfig;

#[cfg(feature = "conv")]
type VaeModel<B> = ConvVAE<B>;
#[cfg(feature = "conv")]
type VaeConfig = ConvVaeConfig;

// --- CONSTANTS ---
const LABEL_PALETTE: [(f32, f32, f32); 10] = [
    (0.12, 0.47, 0.71), // Blue
    (1.00, 0.50, 0.05), // Orange
    (0.17, 0.63, 0.17), // Green
    (0.84, 0.15, 0.16), // Red
    (0.58, 0.40, 0.74), // Purple
    (0.55, 0.34, 0.29), // Brown
    (0.89, 0.47, 0.76), // Pink
    (0.50, 0.50, 0.50), // Gray
    (0.74, 0.74, 0.13), // Olive / Yellow-Green
    (0.09, 0.75, 0.81), // Cyan / Teal
];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 2000)]
    num_samples: usize,
    #[arg(long)]
    xmin: Option<f32>,
    #[arg(long)]
    xmax: Option<f32>,
    #[arg(long)]
    ymin: Option<f32>,
    #[arg(long)]
    ymax: Option<f32>,
    #[arg(long, default_value_t = String::from("single"))]
    plot_type: String,
    #[arg(long, default_value_t = String::from("Latent Space Distribution"))]
    plot_title: String,
}

// --- INTERACTIVE APP STRUCTURE ---

struct InteractiveVaePlot {
    multi_chart: MultiChart,
    is_dragging: bool,
    last_mouse_pos: Point,
}

#[derive(Debug, Clone)]
enum Message {
    EventOccurred(Event),
}

impl InteractiveVaePlot {
    fn new(multi_chart: MultiChart) -> Self {
        Self {
            multi_chart,
            is_dragging: false,
            last_mouse_pos: Point::ORIGIN,
        }
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::EventOccurred(event) => {
                match event {
                    // Start Dragging
                    Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                        self.is_dragging = true;
                    }
                    // Stop Dragging
                    Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                        self.is_dragging = false;
                    }
                    // Rotate on Move
                    Event::Mouse(mouse::Event::CursorMoved { position }) => {
                        if self.is_dragging {
                            let delta_x = position.x - self.last_mouse_pos.x;
                            let delta_y = position.y - self.last_mouse_pos.y;

                            // Sensitivity: 100 pixels = 1 radian
                            let sensitivity = 0.01;

                            // Manually update all 3D charts in the grid
                            // (Since we are outside the lib, we iterate `items`)
                            let mut changed = false;
                            for item in &mut self.multi_chart.items {
                                if let GridItem::Chart3D(chart) = item {
                                    // Mouse X -> Yaw (rotate around Y-axis)
                                    // Mouse Y -> Pitch (rotate around X-axis)
                                    chart.yaw -= (delta_x as f64) * sensitivity;
                                    chart.pitch += (delta_y as f64) * sensitivity;
                                    changed = true;
                                }
                            }

                            // Force redraw if we updated the chart logic
                            // (The `MultiChart` cache needs clearing to know it's dirty)
                            // We need to access the cache clearing method.
                            // If `rotate_all_3d` exists in lib, use that.
                            // Otherwise, assuming `to_element` handles updates implicitly via state
                            // or we trigger a message.
                            // NOTE: Ideally, add `pub fn clear_cache(&mut self)` to MultiChart in lib.
                            // But here, simply changing the data is often enough if the widget logic
                            // checks for changes, OR we just need to ensure the widget is rebuilt.
                            // In plotters-iced, usually clearing the `Cache` is required.

                            // HACK: If you can't access `cache.clear()` from here,
                            // add `self.cache.clear()` inside `MultiChart` methods.
                            if changed {
                                // Call the helper to force redraw
                                self.multi_chart.clear_cache();
                            }
                        }
                        self.last_mouse_pos = position;
                    }
                    _ => {}
                }
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        container(self.multi_chart.to_element())
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        event::listen().map(Message::EventOccurred)
    }
}

// --- MAIN LOGIC ---

fn view_latent_space<B: Backend>(_device: &B::Device, model: VaeModel<B>, args: Args) {
    // 1. Data Loading
    let test_batcher = MnistBatcher::<B>::new();
    let test_loader = DataLoaderBuilder::new(test_batcher)
        .batch_size(args.num_samples)
        .build(MnistDataset::train());

    println!("Starting inference...");
    let (train_images, train_labels) = test_loader.iter().next().expect("Dataset empty");
    let labels: Vec<i32> = train_labels
        .to_data()
        .to_vec()
        .expect("Label extract failed");

    #[cfg(feature = "conv")]
    let train_images = train_images.clone().reshape([0, 1, 28, 28]);

    let (mu, _) = model.encoder.forward(train_images);
    let dims = mu.dims();

    // 2. Extract Latent Vectors
    let x_all: Vec<f32> = mu
        .clone()
        .slice([0..args.num_samples, 0..1])
        .into_data()
        .to_vec()
        .unwrap();
    let y_all: Vec<f32> = mu
        .clone()
        .slice([0..args.num_samples, 1..2])
        .into_data()
        .to_vec()
        .unwrap();
    let z_all: Vec<f32> = if dims[1] >= 3 {
        mu.clone()
            .slice([0..args.num_samples, 2..3])
            .into_data()
            .to_vec()
            .unwrap()
    } else {
        vec![0.0; args.num_samples]
    };

    // --- SINGLE PLOT (INTERACTIVE 3D) ---
    if args.plot_type == "single" {
        let mut chart_series = Vec::new();

        for digit in 0..10 {
            let indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == digit { Some(i) } else { None })
                .collect();

            if indices.is_empty() {
                continue;
            }

            let (r, g, b) = LABEL_PALETTE[digit as usize];
            let digit_color = Some([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);

            let points: Vec<Point3D> = indices
                .iter()
                .map(|&i| Point3D {
                    x: x_all[i],
                    y: y_all[i],
                    z: z_all[i],
                    color: None,
                    radius: None,
                })
                .collect();

            let mut series = Series3D::new(
                format!("Digit {}", digit),
                SeriesStyle::Scatter {
                    radius: 3,
                    filled: true,
                    color: digit_color,
                },
            );
            series.points = points;
            chart_series.push(series);
        }

        let chart_3d = Chart3D::new(args.plot_title.clone())
            .series(chart_series)
            .labels("Latent X", "Latent Y", "Latent Z")
            .view_orientation(0.5, 0.7);

        let multi_chart = MultiChart::new(vec![GridItem::Chart3D(chart_3d)])
            .with_x_space(0)
            .with_y_space(0);

        println!("Launching Interactive 3D Plot. Click and drag to rotate.");

        // Launch Iced Application
        let _ = iced::application(
            "Latendt Space Interactive 3D Plot",
            InteractiveVaePlot::update,
            InteractiveVaePlot::view,
        )
        .subscription(InteractiveVaePlot::subscription)
        .antialiasing(true)
        .run_with(|| (InteractiveVaePlot::new(multi_chart), iced::Task::none()));
    }
    // --- MULTIPLE PLOT (LEGACY 2D GRID) ---
    else if args.plot_type == "multiple" {
        // 3. Create Chart
        let mut items: Vec<GridItem> = Vec::with_capacity(10);

        for digit in 0..10 {
            // 2. Build Separate Series for Digits 0-9
            let mut chart_series = Vec::new();
            // Filter indices where label matches current digit
            let indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == digit { Some(i) } else { None })
                .collect();

            if indices.is_empty() {
                continue;
            }

            // Get the specific color for this digit
            // Get the specific color for this digit
            let (r_f, g_f, b_f) = LABEL_PALETTE[digit as usize];
            let (r, g, b) = (
                (r_f * 255.0) as u8,
                (g_f * 255.0) as u8,
                (b_f * 255.0) as u8,
            );
            let digit_color = Some([r, g, b]);

            // Construct points for this specific digit
            let points: std::collections::VecDeque<PointData> = indices
                .iter()
                .map(|&i| {
                    PointData {
                        x: x_all[i],
                        y: y_all[i],
                        // We can explicitly set point color, or let it inherit from Series default
                        color: None,
                        radius: None,
                    }
                })
                .collect();

            // Create the Series
            let series = Series {
                label: format!("Digit {}", digit), // This will show in the Legend
                points,
                window_len: 0,
                style: SeriesStyle::Scatter {
                    radius: 3,
                    filled: true,
                    color: digit_color,
                },
            };

            // --- Helper closure to apply ranges conditionally ---
            let apply_ranges = |mut chart: SingleChart| -> SingleChart {
                if let (Some(xmin), Some(xmax)) = (args.xmin, args.xmax) {
                    chart = chart.with_x_range(xmin, xmax);
                }
                if let (Some(ymin), Some(ymax)) = (args.ymin, args.ymax) {
                    chart = chart.with_y_range(ymin, ymax);
                }
                chart
            };

            chart_series.push(series);
            let chart = SingleChart::new(args.plot_title.as_str())
                .series(chart_series) // Pass the vector of 10 series
                .x_label("Mu X")
                .y_label("Mu Y");
            let chart = apply_ranges(chart);
            items.push(GridItem::Chart(chart));
        }

        let _chart = MultiChart::new(items).with_x_space(35).with_y_space(35);

        println!("Launching plot with legend...");
        let title_static: &'static str = Box::leak(args.plot_title.into_boxed_str());
        let _ = run_static(title_static, _chart);
        //        let _ = run_static("MNIST VAE Latent Space", _chart);
    }
}

fn main() {
    let args = Args::parse();
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    type MyBackend = Wgpu;
    let device = WgpuDevice::DefaultDevice;

    println!("Loading configuration...");
    let config = VaeConfig::load("vae_model/mnist_vae.config.json").expect("Config missing");

    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load("vae_model/mnist_vae".into(), &device)
        .expect("Weights missing");

    let model = VaeModel::<MyBackend>::new(&config, &device).load_record(record);
    println!("Model loaded.");

    view_latent_space::<MyBackend>(&device, model, args);
}

// #![recursion_limit = "256"]

// // --- IMPORTS ---
// use burn::data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset};
// use burn::tensor::backend::Backend;
// use vae::mnist_data::MnistBatcher;

// // Conditionally import the model structure based on feature flags.
// // This ensures the correct struct is available for the type aliases below.
// #[cfg(feature = "conv")]
// use vae::conv_model::{ConvVAE, ConvVaeConfig};
// #[cfg(feature = "dense")]
// use vae::dense_model::{DenseVAE, DenseVaeConfig};

// // use vae::{MNIST_DIM_X, MNIST_DIM_Y};

// use burn::{
//     config::Config,
//     module::Module,
//     record::{BinFileRecorder, FullPrecisionSettings, Recorder},
// };

// // CLI parsing (included for future extensibility)
// use clap::Parser;
// use tiny_plot_lib::{
//     GridItem, MultiChart, PointData, Series, SeriesStyle, SingleChart, run_static,
// };

// // --- TYPE ALIASES ---

// // Create a type alias for the Model and Config based on the active feature flag.
// // This allows `view_latent_space` to accept the correct type without code duplication.
// #[cfg(feature = "dense")]
// type VaeModel<B> = DenseVAE<B>;
// #[cfg(feature = "dense")]
// type VaeConfig = DenseVaeConfig;

// #[cfg(feature = "conv")]
// type VaeModel<B> = ConvVAE<B>;
// #[cfg(feature = "conv")]
// type VaeConfig = ConvVaeConfig;

// // --- CONSTANTS ---

// /// Semantic Color Palette.
// ///
// /// Maps digits `0-9` to distinct RGB color factors `(R, G, B)` where values range `[0.0, 1.0]`.
// /// Used to tint grayscale images or color scatter points based on their classification label.
// const LABEL_PALETTE: [(f32, f32, f32); 10] = [
//     (0.12, 0.47, 0.71), // Blue
//     (1.00, 0.50, 0.05), // Orange
//     (0.17, 0.63, 0.17), // Green
//     (0.84, 0.15, 0.16), // Red
//     (0.58, 0.40, 0.74), // Purple
//     (0.55, 0.34, 0.29), // Brown
//     (0.89, 0.47, 0.76), // Pink
//     (0.50, 0.50, 0.50), // Gray
//     (0.74, 0.74, 0.13), // Olive / Yellow-Green
//     (0.09, 0.75, 0.81), // Cyan / Teal
// ];
// // const LABEL_PALETTE: [(f32, f32, f32); 10] = [
// //     (0.90, 0.10, 0.10), // Red (slightly darkened for visibility)
// //     (1.00, 0.60, 0.00), // Orange
// //     (1.00, 0.90, 0.10), // Yellow (darkened slightly so it appears on white)
// //     (0.10, 0.70, 0.20), // Green (darker, less neon)
// //     (0.00, 0.80, 0.80), // Cyan/Teal
// //     (0.00, 0.20, 0.60), // Dark Blue (Distinct from Cyan)
// //     (0.40, 0.60, 1.00), // Sky Blue (Distinct from Dark Blue)
// //     (0.60, 0.20, 0.80), // Purple
// //     (0.90, 0.40, 0.70), // Pink/Magenta (Lighter than Purple)
// //     (0.60, 0.40, 0.20), // Brown
// // ];

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// struct Args {
//     /// Number of samples to extract from test set
//     #[arg(long, default_value_t = 2000)]
//     num_samples: usize,

//     /// Min X range for the plot
//     #[arg(long)]
//     xmin: Option<f32>,

//     /// Max X range for the plot
//     #[arg(long)]
//     xmax: Option<f32>,

//     /// Min Y range for the plot
//     #[arg(long)]
//     ymin: Option<f32>,

//     /// Max Y range for the plot
//     #[arg(long)]
//     ymax: Option<f32>,

//     /// type of plot: single or multiple
//     #[arg(long, default_value_t = String::from("single"))]
//     plot_type: String,

//     /// type of plot
//     #[arg(long, default_value_t = String::from("Latent Space Distribution"))]
//     plot_title: String,
// }

// // --- LOGIC ---

// /// Generates a scatter plot of the VAE Latent Space.
// ///
// /// This function performs inference on a batch of test data, extracts the
// /// 2D latent embeddings ($\mu$), and visualizes them.
// ///
// /// # Arguments
// ///
// /// * `_device` - The backend device (required for consistency, though unused in function body).
// /// * `model` - The trained VAE model (Dense or Conv based on feature flags).
// fn view_latent_space<B: Backend>(_device: &B::Device, model: VaeModel<B>, args: Args) {
//     // 1. Data Preparation (Same as before)
//     let test_batcher = MnistBatcher::<B>::new();
//     let test_loader = DataLoaderBuilder::new(test_batcher)
//         .batch_size(args.num_samples)
//         .build(MnistDataset::test());

//     println!("Starting inference...");

//     let (train_images, train_labels) = test_loader.iter().next().expect("Dataset empty");

//     // Extract raw data
//     let labels: Vec<i32> = train_labels
//         .to_data()
//         .to_vec()
//         .expect("Label extract failed");

//     #[cfg(feature = "conv")]
//     // Reshape for Conv Network: [B, 784] -> [B, 1, 28, 28]
//     // Convolutional layers require (Batch, Channels, Height, Width) format.
//     let train_images = train_images.clone().reshape([0, 1, 28, 28]);
//     let (mu, _) = model.encoder.forward(train_images);

//     let x_all: Vec<f32> = mu
//         .clone()
//         .slice([0..args.num_samples, 0..1])
//         .into_data()
//         .to_vec()
//         .unwrap();
//     let y_all: Vec<f32> = mu
//         .clone()
//         .slice([0..args.num_samples, 1..2])
//         .into_data()
//         .to_vec()
//         .unwrap();
//     // --- Helper closure to apply ranges conditionally ---
//     let apply_ranges = |mut chart: SingleChart| -> SingleChart {
//         if let (Some(xmin), Some(xmax)) = (args.xmin, args.xmax) {
//             chart = chart.with_x_range(xmin, xmax);
//         }
//         if let (Some(ymin), Some(ymax)) = (args.ymin, args.ymax) {
//             chart = chart.with_y_range(ymin, ymax);
//         }
//         chart
//     };
//     if args.plot_type == "single" {
//         // 2. Build Separate Series for Digits 0-9
//         let mut chart_series = Vec::new();

//         for digit in 0..10 {
//             // Filter indices where label matches current digit
//             let indices: Vec<usize> = labels
//                 .iter()
//                 .enumerate()
//                 .filter_map(|(i, &l)| if l == digit { Some(i) } else { None })
//                 .collect();

//             if indices.is_empty() {
//                 continue;
//             }

//             // Get the specific color for this digit
//             // Get the specific color for this digit
//             let (r_f, g_f, b_f) = LABEL_PALETTE[digit as usize];
//             let (r, g, b) = (
//                 (r_f * 255.0) as u8,
//                 (g_f * 255.0) as u8,
//                 (b_f * 255.0) as u8,
//             );
//             let digit_color = Some([r, g, b]);

//             // Construct points for this specific digit
//             let points: std::collections::VecDeque<PointData> = indices
//                 .iter()
//                 .map(|&i| {
//                     PointData {
//                         x: x_all[i],
//                         y: y_all[i],
//                         // We can explicitly set point color, or let it inherit from Series default
//                         color: None,
//                         radius: None,
//                     }
//                 })
//                 .collect();

//             // Create the Series
//             let series = Series {
//                 label: format!("Digit {}", digit), // This will show in the Legend
//                 points,
//                 window_len: 0,
//                 style: SeriesStyle::Scatter {
//                     radius: 3,
//                     filled: true,
//                     color: digit_color,
//                 },
//             };

//             chart_series.push(series);
//         }

//         // 3. Create Chart
//         let mut items: Vec<GridItem> = Vec::with_capacity(10);

//         let mut chart = SingleChart::new(args.plot_title)
//             .series(chart_series) // Pass the vector of 10 series
//             .x_label("Mu X")
//             .y_label("Mu Y");
//         chart = apply_ranges(chart);

//         items.push(GridItem::Chart(chart));

//         let chart = MultiChart::new(items).with_x_space(55).with_y_space(55);

//         println!("Launching plot with legend...");
//         let _ = run_static("MNIST VAE Latent Space", chart);
//     }
//     ////////////////// second run
//     else if args.plot_type == "multiple" {
//         // 3. Create Chart
//         let mut items: Vec<GridItem> = Vec::with_capacity(10);

//         for digit in 0..10 {
//             // 2. Build Separate Series for Digits 0-9
//             let mut chart_series = Vec::new();
//             // Filter indices where label matches current digit
//             let indices: Vec<usize> = labels
//                 .iter()
//                 .enumerate()
//                 .filter_map(|(i, &l)| if l == digit { Some(i) } else { None })
//                 .collect();

//             if indices.is_empty() {
//                 continue;
//             }

//             // Get the specific color for this digit
//             // Get the specific color for this digit
//             let (r_f, g_f, b_f) = LABEL_PALETTE[digit as usize];
//             let (r, g, b) = (
//                 (r_f * 255.0) as u8,
//                 (g_f * 255.0) as u8,
//                 (b_f * 255.0) as u8,
//             );
//             let digit_color = Some([r, g, b]);

//             // Construct points for this specific digit
//             let points: std::collections::VecDeque<PointData> = indices
//                 .iter()
//                 .map(|&i| {
//                     PointData {
//                         x: x_all[i],
//                         y: y_all[i],
//                         // We can explicitly set point color, or let it inherit from Series default
//                         color: None,
//                         radius: None,
//                     }
//                 })
//                 .collect();

//             // Create the Series
//             let series = Series {
//                 label: format!("Digit {}", digit), // This will show in the Legend
//                 points,
//                 window_len: 0,
//                 style: SeriesStyle::Scatter {
//                     radius: 3,
//                     filled: true,
//                     color: digit_color,
//                 },
//             };

//             chart_series.push(series);
//             let chart = SingleChart::new(args.plot_title.as_str())
//                 .series(chart_series) // Pass the vector of 10 series
//                 .x_label("Mu X")
//                 .y_label("Mu Y");
//             let chart = apply_ranges(chart);
//             items.push(GridItem::Chart(chart));
//         }

//         let _chart = MultiChart::new(items).with_x_space(35).with_y_space(35);

//         println!("Launching plot with legend...");
//         let title_static: &'static str = Box::leak(args.plot_title.into_boxed_str());
//         let _ = run_static(title_static, _chart);
// //        let _ = run_static("MNIST VAE Latent Space", _chart);
//     }
// }

// /// Entry point for the application.
// ///
// /// Sets up the WGPU backend, loads the configuration and model weights,
// /// and triggers the latent space visualization.
// fn main() {
//     // 1. Parse CLI Arguments
//     let args = Args::parse();

//     // 1. Initialize Backend
//     // WGPU is used for cross-platform hardware acceleration (Metal, Vulkan, DX12).
//     use burn::backend::wgpu::{Wgpu, WgpuDevice};

//     // Define the backend type alias
//     type MyBackend = Wgpu;

//     // Automatically select the best available graphics device
//     let device = WgpuDevice::DefaultDevice;

//     println!("Loading configuration...");

//     // 2. Load Model Configuration
//     // The config file must match the structure defined by the enabled feature (dense vs conv).
//     let config = VaeConfig::load("vae_model/mnist_vae.config.json")
//         .expect("Config file not found. Ensure 'mnist_vae.config.json' exists in 'vae_model/'");

//     println!("Model config loaded: {}", config);

//     // 3. Load Pre-trained Model Weights
//     // The recorder handles binary deserialization of the weight tensors.
//     let record = BinFileRecorder::<FullPrecisionSettings>::new()
//         .load("vae_model/mnist_vae".into(), &device)
//         .expect("Model weights file not found. Did you run the training script?");

//     // 4. Instantiate and Load Model
//     // We create the model using the config and immediately load the recorded state.
//     let model = VaeModel::<MyBackend>::new(&config, &device).load_record(record);

//     println!("Model loaded successfully.");

//     // 5. Execute Visualization
//     view_latent_space::<MyBackend>(&device, model, args);
// }
