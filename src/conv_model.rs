use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    tensor::{Distribution, Tensor, backend::Backend},
};

// --- CONFIG ---

/// Configuration for a convolutional Variational Autoencoder (VAE).
///
/// This defines latent dimensionality, convolutional channel depth,
/// training parameters, and compatibility settings for input shape.
#[derive(Config, Debug)]
pub struct ConvVaeConfig {
    /// Flattened input dimension (e.g. 28×28=784).
    /// Kept for compatibility with dense-training utilities.
    #[config(default = 784)]
    pub input_dim: usize,
    /// Dimensionality of the latent vector.
    #[config(default = 20)]
    pub latent_dim: usize,
    /// Learning rate used during optimization.
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Number of training epochs.
    #[config(default = 25)]
    pub num_epochs: usize,
    /// Number of samples per training batch.
    #[config(default = 128)]
    pub batch_size: usize,
    /// Base convolution channel depth (first conv layer = `base_channels`).
    #[config(default = 32)]
    pub base_channels: usize,
}

// --- CONV ENCODER ---

/// Convolutional encoder for the VAE.
///
/// Structure:
/// - Two conv layers downsampling 28×28 → 14×14 → 7×7
/// - Flatten to a linear vector
/// - Produce `mu` and `logvar` for the latent distribution
#[derive(Module, Debug)]
pub struct ConvEncoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
    activation: Relu,
    /// Flattened feature dimension after final conv layer.
    flattened_dim: usize,
}

impl<B: Backend> ConvEncoder<B> {
    /// Construct a new convolutional encoder.
    ///
    /// # Arguments
    /// * `config` – VAE configuration (latent size, channels, etc.)
    /// * `device` – Backend device where parameters are allocated
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        let c = config.base_channels;

        // Layer 1:
        // Input:  [1, 28, 28]
        // Output: [32, 14, 14]
        let conv1 = Conv2dConfig::new([1, c], [3, 3])
            .with_stride([2, 2]) // Downsample by 2
            .with_padding(PaddingConfig2d::Explicit(1, 1)) // Keep spatial center alignment
            .init(device);

        // Layer 2:
        // Input:  [32, 14, 14]
        // Output: [64, 7, 7]
        let conv2 = Conv2dConfig::new([c, c * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // After two downsampling steps: shape = (channels=64) * 7 * 7
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

    /// Forward pass for the encoder.
    ///
    /// # Arguments
    /// * `x` – Input batch of shape `[Batch, 1, 28, 28]`
    ///
    /// # Returns
    /// `(mu, logvar)` – tensors of shape `[Batch, latent_dim]`
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // First convolution + activation
        let x = self.activation.forward(self.conv1.forward(x));
        // Second convolution + activation
        let x = self.activation.forward(self.conv2.forward(x));

        // Flatten feature maps from [B, C, 7, 7] → [B, 3136]
        let x = x.flatten(1, 3);

        // Latent distribution parameters
        let mu = self.fc_mu.forward(x.clone());
        let sigma = self.fc_logvar.forward(x);

        (mu, sigma)
    }
}

// --- CONV DECODER ---

/// Convolutional decoder for the VAE.
///
/// Structure:
/// - Linear layer expands latent vector into (channels×7×7)
/// - Two transposed convolutions upsample 7×7 → 14×14 → 28×28
/// - Final sigmoid to produce image-like output
#[derive(Module, Debug)]
pub struct ConvDecoder<B: Backend> {
    fc_initial: Linear<B>,
    convt1: ConvTranspose2d<B>,
    convt2: ConvTranspose2d<B>,
    activation: Relu,
    /// Base channel depth for reconstructing shapes.
    base_channels: usize,
}

impl<B: Backend> ConvDecoder<B> {
    /// Construct a new convolutional decoder.
    ///
    /// # Arguments
    /// * `config` – VAE configuration
    /// * `device` – Backend device
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        let c = config.base_channels;

        // Flattened feature dimension from encoder
        let flattened_dim = (c * 2) * 7 * 7;

        // Expand latent vector z → initial feature map
        let fc_initial = LinearConfig::new(config.latent_dim, flattened_dim).init(device);

        // Transposed convolution 1:
        // Input:  [64, 7, 7]
        // Output: [32, 14, 14]
        let convt1 = ConvTranspose2dConfig::new([c * 2, c], [3, 3])
            .with_stride([2, 2])
            // Note: Uses raw array (not PaddingConfig2d) for ConvTranspose
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);

        // Transposed convolution 2:
        // Input:  [32, 14, 14]
        // Output: [1, 28, 28]
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

    /// Forward pass for the decoder.
    ///
    /// # Arguments
    /// * `z` – Latent sample, shape `[Batch, latent_dim]`
    ///
    /// # Returns
    /// Reconstructed images with shape `[Batch, 1, 28, 28]`
    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 4> {
        // Expand latent vector to flattened conv feature map
        let x = self.activation.forward(self.fc_initial.forward(z));

        // Unflatten to [Batch, Channels=64, 7, 7]
        let x = x.reshape([0, (self.base_channels as i32 * 2), 7, 7]);

        // Upsample → 14×14
        let x = self.activation.forward(self.convt1.forward(x));

        // Final upsample → 28×28 + sigmoid for pixel range
        burn::tensor::activation::sigmoid(self.convt2.forward(x))
    }
}

// --- CONV VAE MODULE ---

/// Full convolutional VAE composed of:
/// - ConvEncoder
/// - ConvDecoder
///
/// Includes reparameterization and end-to-end forward pass.
#[derive(Module, Debug)]
pub struct ConvVAE<B: Backend> {
    /// Convolutional encoder producing latent mean and logvar.
    pub encoder: ConvEncoder<B>,
    /// Convolutional decoder generating reconstructed images.
    pub decoder: ConvDecoder<B>,
}

impl<B: Backend> ConvVAE<B> {
    /// Construct a convolutional VAE from configuration and device.
    pub fn new(config: &ConvVaeConfig, device: &B::Device) -> Self {
        Self {
            encoder: ConvEncoder::new(config, device),
            decoder: ConvDecoder::new(config, device),
        }
    }

    /// Apply the reparameterization trick.
    ///
    /// Computes:
    /// ```
    /// std = exp(0.5 * logvar)
    /// eps ~ N(0, 1)
    /// z = mu + eps * std
    /// ```
    ///
    /// # Arguments
    /// * `mu` – Mean of the approximate posterior
    /// * `sigma` – Log-variance of the approximate posterior
    ///
    /// # Returns
    /// Latent sample `z` of shape `[Batch, latent_dim]`
    pub fn reparameterize(&self, mu: Tensor<B, 2>, sigma: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = sigma.mul_scalar(0.5).exp();
        let eps = Tensor::random_like(&std, Distribution::Normal(0.0, 1.0));
        mu + eps * std
    }

    /// Full VAE forward pass.
    ///
    /// # Arguments
    /// * `x` – Input images `[Batch, 1, 28, 28]`
    ///
    /// # Returns
    /// `(recon_flat, mu, logvar)`
    ///
    /// * `recon_flat`: Reconstructed images flattened to `[Batch, 784]`
    /// * `mu`: Latent mean
    /// * `logvar`: Latent log-variance
    ///
    /// The output is flattened because many training utilities expect
    /// a vector input for reconstruction loss.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Encode to latent distribution parameters
        let (mu, sigma) = self.encoder.forward(x);

        // Reparameterization trick
        let z = self.reparameterize(mu.clone(), sigma.clone());

        // Decode latent vector → image
        let recon_img = self.decoder.forward(z);

        // Flatten output back to vector shape: [Batch, 1, 28, 28] → [Batch, 784]
        let recon_flat = recon_img.flatten(1, 3);

        (recon_flat, mu, sigma)
    }
}
