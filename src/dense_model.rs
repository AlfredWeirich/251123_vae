use crate::reparameterize;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, activation, backend::Backend},
};

// --- CONFIG ---

/// Configuration for the Dense (Fully Connected) Variational Autoencoder.
///
/// This struct defines the topology of the neural network, including input dimensions,
/// the hierarchy of hidden layers, and the size of the latent bottleneck.
#[derive(Config, Debug)]
pub struct DenseVaeConfig {
    /// The size of the input feature vector (e.g., 784 for flattened MNIST images).
    #[config(default = 784)]
    pub input_dim: usize,

    /// A list defining the topology of the hidden layers in the Encoder.
    ///
    /// The Decoder will automatically construct the symmetric reverse of this topology.
    /// Example: `vec![512, 256]` results in:
    /// *   **Encoder:** Input -> 512 -> 256 -> Latent
    /// *   **Decoder:** Latent -> 256 -> 512 -> Input
    #[config(default = "vec![32,12]")]
    pub hidden_dims: Vec<usize>,

    /// The dimension of the latent space (z).
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
}

// --- ENCODER ---

/// The Probabilistic Encoder $q_\phi(z|x)$.
///
/// Maps input data to the parameters of the approximate posterior distribution
/// (mean and log-variance). It supports a dynamic number of hidden layers
/// based on the configuration.
#[derive(Module, Debug)]
pub struct DenseEncoder<B: Backend> {
    /// Dynamic stack of fully connected hidden layers.
    layers: Vec<Linear<B>>,
    /// Linear head to predict the mean ($\mu$) of the latent distribution.
    fc_mu: Linear<B>,
    /// Linear head to predict the log-variance ($\log \sigma^2$) of the latent distribution.
    fc_logvar: Linear<B>,
    /// Activation function applied after every hidden layer.
    activation: Relu,
}

impl<B: Backend> DenseEncoder<B> {
    /// Constructs a new `DenseEncoder` based on the provided configuration.
    ///
    /// This initializes weights and biases on the specified `device`.
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = config.input_dim;

        // Dynamically build the hidden layers stack based on config topology.
        for &dim in &config.hidden_dims {
            layers.push(LinearConfig::new(current_dim, dim).init(device));
            current_dim = dim;
        }

        // The output of the last hidden layer branches into two separate linear layers
        // for the Gaussian parameters.
        let fc_mu = LinearConfig::new(current_dim, config.latent_dim).init(device);
        let fc_logvar = LinearConfig::new(current_dim, config.latent_dim).init(device);

        Self {
            layers,
            fc_mu,
            fc_logvar,
            activation: Relu::new(),
        }
    }

    /// Performs the forward pass of the encoder.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `(Batch, Input_Dim)`.
    ///
    /// # Returns
    /// A tuple `(mu, logvar)` where:
    /// * `mu` - The mean of the latent distribution. Shape: `(Batch, Latent_Dim)`.
    /// * `logvar` - The log-variance of the latent distribution. Shape: `(Batch, Latent_Dim)`.
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = x;

        // Propagate through dynamic hidden layers
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Split into two heads
        let mu = self.fc_mu.forward(x.clone());
        let logvar = self.fc_logvar.forward(x);

        (mu, logvar)
    }
}

// --- DECODER ---

/// The Probabilistic Decoder $p_\theta(x|z)$.
///
/// Maps samples from the latent space back to the data space (reconstruction).
/// The architecture is symmetric to the Encoder.
#[derive(Module, Debug)]
pub struct DenseDecoder<B: Backend> {
    /// Dynamic stack of fully connected hidden layers (reversed topology).
    layers: Vec<Linear<B>>,
    /// Final projection layer to the original input dimension.
    output_layer: Linear<B>,
    /// Activation function applied after hidden layers.
    activation: Relu,
}

impl<B: Backend> DenseDecoder<B> {
    /// Constructs a new `DenseDecoder`.
    ///
    /// Iterates through `hidden_dims` in reverse to create a mirror image of the encoder.
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = config.latent_dim;

        // Iterate in REVERSE order to build a symmetric decoder.
        // E.g., if Encoder is 784 -> 512 -> 256 -> Latent
        // Decoder becomes Latent -> 256 -> 512 -> 784
        for &dim in config.hidden_dims.iter().rev() {
            layers.push(LinearConfig::new(current_dim, dim).init(device));
            current_dim = dim;
        }

        // Final mapping to original input dimension
        let output_layer = LinearConfig::new(current_dim, config.input_dim).init(device);

        Self {
            layers,
            output_layer,
            activation: Relu::new(),
        }
    }

    /// Performs the forward pass of the decoder.
    ///
    /// # Arguments
    /// * `z` - Latent samples. Shape: `(Batch, Latent_Dim)`.
    ///
    /// # Returns
    /// * `recon_x` - The reconstructed input. Shape: `(Batch, Input_Dim)`.
    ///   Values are squashed to `[0, 1]` via Sigmoid (interpretable as probabilities
    ///   or pixel intensities).
    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = z;

        // Pass through dynamic hidden layers
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Final layer + Sigmoid for reconstruction
        // Sigmoid is standard for MNIST to ensure outputs are in range [0, 1]
        let x = self.output_layer.forward(x);
        activation::sigmoid(x)
    }
}

// --- FULL VAE ---

/// The complete Variational Autoencoder module.
///
/// Combines the `DenseEncoder` and `DenseDecoder` to form the full pipeline:
/// Input $\to$ Encoder $\to$ Reparameterization $\to$ Decoder $\to$ Reconstruction.
#[derive(Module, Debug)]
pub struct DenseVAE<B: Backend> {
    pub encoder: DenseEncoder<B>,
    pub decoder: DenseDecoder<B>,
}

impl<B: Backend> DenseVAE<B> {
    /// Initializes the VAE model components on the specified device.
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        Self {
            encoder: DenseEncoder::new(config, device),
            decoder: DenseDecoder::new(config, device),
        }
    }

    /// The full forward pass of the VAE.
    ///
    /// # Steps
    /// 1. **Encode**: Map input `x` to `mu` and `logvar`.
    /// 2. **Reparameterize**: Sample `z = mu + sigma * epsilon`.
    /// 3. **Decode**: Reconstruct `recon_x` from `z`.
    ///
    /// # Arguments
    /// * `x` - Input tensor. Shape: `(Batch, Input_Dim)`.
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. `recon_x`: Reconstructed input. Shape: `(Batch, Input_Dim)`.
    /// 2. `mu`: Latent mean. Shape: `(Batch, Latent_Dim)`.
    /// 3. `logvar`: Latent log-variance. Shape: `(Batch, Latent_Dim)`.
    ///
    /// These three tensors are required to compute the VAE Loss (ELBO).
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mu, logvar) = self.encoder.forward(x);
        let z = reparameterize(mu.clone(), logvar.clone());
        let recon_x = self.decoder.forward(z);
        (recon_x, mu, logvar)
    }
}
