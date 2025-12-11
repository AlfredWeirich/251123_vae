use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use std::marker::PhantomData;

// --- CONSTANTS ---
const IMAGE_H: usize = 28;
const IMAGE_W: usize = 28;
const IMAGE_FLAT_DIM: usize = IMAGE_H * IMAGE_W;
const NORMALIZATION_FACTOR: f32 = 255.0;

// --- Data Handling ---

/// A Batcher implementation for the MNIST dataset.
///
/// The responsibility of the `MnistBatcher` is to transform a vector of raw `MnistItem`s
/// (loaded from disk/CPU) into tensors allocated on the specific backend device (GPU/TPU).
///
/// # Key Operations
/// 1. **Flattening**: Converts 2D images (28 x 28) into 1D vectors (784).
/// 2. **Normalization**: Scales pixel intensity from integer `[0, 255]` to float `[0.0, 1.0]`.
/// 3. **Device Transfer**: moves data to the active computation device.
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    /// Zero-sized marker to associate the Batcher with a specific Backend `B`
    /// without actually storing backend data.
    _b: PhantomData<B>,
}

impl<B: Backend> MnistBatcher<B> {
    /// Creates a new instance of the MNIST batcher.
    pub fn new() -> Self {
        Self { _b: PhantomData }
    }
}

impl<B: Backend> Default for MnistBatcher<B> {
    /// Returns a default batcher instance.
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Batcher<B, MnistItem, (Tensor<B, 2>, Tensor<B, 1, Int>)> for MnistBatcher<B> {
    /// Processes a list of items into a batch of tensors.
    ///
    /// # Arguments
    /// * `items` - A vector of raw MNIST items (image + label) loaded by the Dataset iterator.
    /// * `device` - The target device (e.g., `WgpuDevice`, `CudaDevice`) where tensors will be allocated.
    ///
    /// # Returns
    /// A tuple `(images, labels)`:
    /// * **images**: Float tensor of shape `[batch_size, 784]`.
    ///   * Note: The VAE/CNN model may need to reshape this to `[batch_size, 1, 28, 28]`.
    /// * **labels**: Int tensor of shape `[batch_size]`.
    ///   * Note: Even for unsupervised VAEs, labels are returned here to allow for
    ///     validation steps or semantic visualization (coloring by digit).
    fn batch(
        &self,
        items: Vec<MnistItem>,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        let batch_size = items.len();

        // --- 1. Process Images ---

        // We use a single flat vector to collect all pixels from all images in the batch.
        // This minimizes allocations compared to creating vectors per image.
        //
        // Transformation:
        // u8 (0..255) -> f32 (0.0..1.0)
        let images_flattened: Vec<f32> = items
            .iter()
            .flat_map(|item| {
                item.image
                    .iter() // Iterate rows
                    .flat_map(|row| row.iter()) // Iterate pixels
                    .map(|&pixel| pixel / NORMALIZATION_FACTOR) // Normalize
            })
            .collect();

        // Create the tensor from the flattened data.
        // Shape: [Batch_Size, 784]
        let images = Tensor::from_floats(
            TensorData::new(images_flattened, vec![batch_size, IMAGE_FLAT_DIM]),
            device,
        );

        // --- 2. Process Labels ---

        // Extract labels and cast to i64 (Burn's standard integer type for tensors).
        let labels_data: Vec<i64> = items.iter().map(|item| item.label as i64).collect();

        // Shape: [Batch_Size]
        let labels = Tensor::from_ints(TensorData::new(labels_data, vec![batch_size]), device);

        (images, labels)
    }
}
