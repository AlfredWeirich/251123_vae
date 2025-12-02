use crate::reparameterize;
use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    tensor::{Tensor, backend::Backend},
};

// --- CONFIG ---

/// Configuration for the Convolutional Variational Autoencoder (VAE).
///
/// This struct defines the hyperparameters for training and architecture construction.
/// It includes latent space dimensions, channel depths for convolutional layers,
/// and batching parameters.
#[derive(Config, Debug)]
pub struct ConvVaeConfig {
    /// The size of the flattened input vector (e.g., 28×28 = 784).
    /// Used primarily for compatibility with loss functions expecting flat vectors.
    #[config(default = 784)]
    pub input_dim: usize,

    /// Dimensionality of the latent space ($z$).
    #[config(default = 20)]
    pub latent_dim: usize,

    /// Learning rate for the optimizer.
    #[config(default = 1e-3)]
    pub learning_rate: f64,

    /// Total number of training epochs.
    #[config(default = 25)]
    pub num_epochs: usize,

    /// Mini-batch size for training and inference.
    #[config(default = 128)]
    pub batch_size: usize,

    /// The number of channels in the first convolutional layer.
    /// Subsequent layers typically double this depth (e.g., 32 -> 64).
    #[config(default = 32)]
    pub base_channels: usize,
}

// --- CONV ENCODER ---

/// The Convolutional Encoder Network.
///
/// Compresses input images into a low-dimensional latent space using
/// a series of downsampling convolutional layers.
///
/// # Architecture
/// 1. **Conv2d**: 28×28 → 14×14 (Stride 2)
/// 2. **Conv2d**: 14×14 → 7×7 (Stride 2)
/// 3. **Flatten**: 7×7×Channels → Vector
/// 4. **Linear**: Produces $\mu$ and $\log\sigma^2$
#[derive(Module, Debug)]
pub struct ConvEncoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
    activation: Relu,
    /// The size of the flattened feature map after the final convolution.
    /// Used to initialize the linear layers.
    flattened_dim: usize,
}

impl<B: Backend> ConvEncoder<B> {
    /// Constructs a new `ConvEncoder`.
    ///
    /// Initializes weights and computes feature map dimensions.
    ///
    /// # Arguments
    /// * `config` - The configuration containing channel depths and latent size.
    /// * `device` - The backend device for tensor allocation.
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        let c = config.base_channels;

        // Layer 1: Downsample 28x28 -> 14x14
        // Input channels: 1 (Grayscale), Output: base_channels (e.g., 32)
        let conv1 = Conv2dConfig::new([1, c], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)) // Keeps alignment
            .init(device);

        // Layer 2: Downsample 14x14 -> 7x7
        // Input: base_channels, Output: base_channels * 2 (e.g., 64)
        let conv2 = Conv2dConfig::new([c, c * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Calculate flattened dimension: (Channels * Height * Width)
        // (c * 2) channels * 7 height * 7 width
        let flattened_dim = (c * 2) * 7 * 7;

        let fc_mu = LinearConfig::new(flattened_dim, config.latent_dim).init(device);
        let fc_logvar = LinearConfig::new(flattened_dim, config.latent_dim).init(device);

        Self {
            conv1,
            conv2,
            fc_mu,
            fc_logvar,
            activation: Relu::new(),
            flattened_dim,
        }
    }

    /// Performs the forward pass of the encoder.
    ///
    /// # Arguments
    /// * `x` - Input image tensor. Shape: `(Batch, 1, 28, 28)`.
    ///
    /// # Returns
    /// A tuple `(mu, logvar)`:
    /// * `mu`: Latent mean. Shape: `(Batch, Latent_Dim)`.
    /// * `logvar`: Latent log-variance. Shape: `(Batch, Latent_Dim)`.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Convolutions + ReLU
        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.activation.forward(self.conv2.forward(x));

        // Flatten: [Batch, Channels, Height, Width] -> [Batch, Flattened_Dim]
        let x = x.flatten(1, 3);

        // Project to latent parameters
        let mu = self.fc_mu.forward(x.clone());
        let sigma = self.fc_logvar.forward(x);

        (mu, sigma)
    }
}

// --- CONV DECODER ---

/// The Convolutional Decoder Network.
///
/// Reconstructs images from the latent vectors using Transposed Convolutions (Upsampling).
///
/// # Architecture
/// 1. **Linear**: Latent Vector → Flattened Feature Map (7×7×64)
/// 2. **Reshape**: Unflatten to 4D tensor
/// 3. **ConvTranspose2d**: 7×7 → 14×14
/// 4. **ConvTranspose2d**: 14×14 → 28×28
/// 5. **Sigmoid**: Squash outputs to [0, 1]
#[derive(Module, Debug)]
pub struct ConvDecoder<B: Backend> {
    fc_initial: Linear<B>,
    convt1: ConvTranspose2d<B>,
    convt2: ConvTranspose2d<B>,
    activation: Relu,
    /// Base channel depth used to reshape the initial linear output.
    base_channels: usize,
}

impl<B: Backend> ConvDecoder<B> {
    /// Constructs a new `ConvDecoder`.
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        let c = config.base_channels;

        // Must match encoder output dimensions: (c * 2) * 7 * 7
        let flattened_dim = (c * 2) * 7 * 7;

        // Initial expansion from Latent Space
        let fc_initial = LinearConfig::new(config.latent_dim, flattened_dim).init(device);

        // Layer 1: Upsample 7x7 -> 14x14
        // Input: 64 channels, Output: 32 channels
        let convt1 = ConvTranspose2dConfig::new([c * 2, c], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1]) // Crucial for correct output size
            .init(device);

        // Layer 2: Upsample 14x14 -> 28x28
        // Input: 32 channels, Output: 1 channel (Grayscale)
        let convt2 = ConvTranspose2dConfig::new([c, 1], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);

        Self {
            fc_initial,
            convt1,
            convt2,
            activation: Relu::new(),
            base_channels: c,
        }
    }

    /// Performs the forward pass of the decoder.
    ///
    /// # Arguments
    /// * `z` - Latent samples. Shape: `(Batch, Latent_Dim)`.
    ///
    /// # Returns
    /// * `recon_img` - Reconstructed images. Shape: `(Batch, 1, 28, 28)`.
    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 4> {
        // Expand latent vector
        let x = self.activation.forward(self.fc_initial.forward(z));

        // Unflatten/Reshape to [Batch, Channels, Height, Width]
        // Note: Reshape dimension must match the calculated flattened size.
        let x = x.reshape([0, (self.base_channels as i32 * 2), 7, 7]);

        // Upsampling layers
        let x = self.activation.forward(self.convt1.forward(x));

        // Final layer: No ReLU here, just Sigmoid for pixel range [0, 1]
        burn::tensor::activation::sigmoid(self.convt2.forward(x))
    }
}

// --- CONV VAE MODULE ---

/// The complete Convolutional Variational Autoencoder.
///
/// Wraps the `ConvEncoder` and `ConvDecoder` and implements the reparameterization trick.
#[derive(Module, Debug)]
pub struct ConvVAE<B: Backend> {
    pub encoder: ConvEncoder<B>,
    pub decoder: ConvDecoder<B>,
}

impl<B: Backend> ConvVAE<B> {
    /// Constructs the full VAE model.
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        Self {
            encoder: ConvEncoder::new(config, device),
            decoder: ConvDecoder::new(config, device),
        }
    }

    /// The full forward pass of the VAE.
    ///
    /// # Arguments
    /// * `x` - Input image batch. Shape: `(Batch, 1, 28, 28)`.
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. `recon_flat`: Reconstructed images, FLATTENED to `(Batch, 784)`.
    ///    This flattening is done for compatibility with standard MSE/BCE loss functions.
    /// 2. `mu`: Latent mean. Shape: `(Batch, Latent_Dim)`.
    /// 3. `logvar`: Latent log-variance. Shape: `(Batch, Latent_Dim)`.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Encode
        let (mu, sigma) = self.encoder.forward(x);

        // Reparameterize: z = mu + sigma * epsilon
        let z = reparameterize(mu.clone(), sigma.clone());

        // Decode
        let recon_img = self.decoder.forward(z);

        // Flatten output for loss calculation: [Batch, 1, 28, 28] -> [Batch, 784]
        let recon_flat = recon_img.flatten(1, 3);

        (recon_flat, mu, sigma)
    }
}
