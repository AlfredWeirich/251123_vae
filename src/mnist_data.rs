use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use std::marker::PhantomData;

// --- Data Handling ---
/// A batcher for MNIST data that converts a list of `MnistItem`s
/// (each containing a 28×28 grayscale image) into a normalized tensor
/// suitable for feeding into a neural network.
///
/// This struct is generic over any Burn backend. It does not store data
/// itself; instead, it operates on incoming batches and prepares them
/// for model consumption.
///
/// # Normalization
/// Pixel values are converted from `u8` (0–255) to `f32` in the range:
/// ```text
/// pixel_normalized = pixel / 255.0
/// ```
///
/// # Output Shape
/// Produces a tensor of shape:
/// ```text
/// [batch_size, 784]
/// ```
/// where each image is flattened row-major.
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    /// Phantom marker indicating which backend this batcher works with.
    _b: PhantomData<B>,
}

impl<B: Backend> MnistBatcher<B> {
        /// Creates a new MNIST batcher instance.
    ///
    /// Typically used when constructing a dataset loader for MNIST.
    pub fn new() -> Self {
        Self { _b: PhantomData }
    }
}

impl<B: Backend> Default for MnistBatcher<B> {
        /// Creates a default MNIST batcher.
    ///
    /// Equivalent to calling [`MnistBatcher::new()`].
        fn default() -> Self {
        Self::new()
    }
}


impl<B: Backend> Batcher<B, MnistItem, (Tensor<B, 2>, Tensor<B, 1, Int>)> for MnistBatcher<B> {
        /// Converts a vector of MNIST items into a single batched tensor.
    ///
    /// # Arguments
    /// * `items` — A vector of `MnistItem`s, each containing a 28×28 image grid.
    /// * `device` — Backend device the resulting tensor should be moved to.
    ///
    /// # Returns
    /// A tensor of shape `[batch_size, 784]` containing normalized pixel data.
    ///
    /// # Process
    /// - Iterates through the batch of images
    /// - Flattens each image (28×28 → 784)
    /// - Normalizes pixel intensities from 0–255 into 0.0–1.0
    /// - Packs into a tensor ready for training or inference
    fn batch(
        &self,
        items: Vec<MnistItem>,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        let batch_size = items.len();

        // --- Process Images ---
        // Flatten 28x28 images and normalize to [0.0, 1.0]
        let images_flattened: Vec<f32> = items
            .iter()
            .flat_map(|item| {
                item.image
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|&pixel| pixel as f32 / 255.0)
            })
            .collect();

        let images = Tensor::from_floats(
            TensorData::new(images_flattened, vec![batch_size, 784]),
            device,
        );

        // --- Process Labels ---
        // Extract labels and convert to i64 (standard integer type for Burn)
        let labels_data: Vec<i64> = items.iter().map(|item| item.label as i64).collect();

        let labels = Tensor::from_ints(TensorData::new(labels_data, vec![batch_size]), device);

        (images, labels)
    }
}


