use crate::reparameterize;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, activation, backend::Backend},
};

// --- CONFIG ---

#[derive(Config, Debug)]
pub struct DenseVaeConfig {
    #[config(default = 784)]
    pub input_dim: usize,

    /// List of hidden layer sizes.
    /// e.g., vec![512, 256] creates: Input -> 512 -> 256 -> Latent
    #[config(default = "vec![512, 256, 64]")]
    pub hidden_dims: Vec<usize>,

    #[config(default = 20)]
    pub latent_dim: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 25)]
    pub num_epochs: usize,
    #[config(default = 128)]
    pub batch_size: usize,
}

// --- ENCODER ---

#[derive(Module, Debug)]
pub struct DenseEncoder<B: Backend> {
    // We use a vector of Linear layers for the dynamic hidden stack
    layers: Vec<Linear<B>>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
    activation: Relu,
}

impl<B: Backend> DenseEncoder<B> {
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = config.input_dim;

        // Create a linear layer for every dimension in hidden_dims
        for &dim in &config.hidden_dims {
            layers.push(LinearConfig::new(current_dim, dim).init(device));
            current_dim = dim;
        }

        // The final hidden dim connects to the latent distribution params
        let fc_mu = LinearConfig::new(current_dim, config.latent_dim).init(device);
        let fc_logvar = LinearConfig::new(current_dim, config.latent_dim).init(device);

        Self {
            layers,
            fc_mu,
            fc_logvar,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = x;

        // Pass through dynamic hidden layers
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let mu = self.fc_mu.forward(x.clone());
        let logvar = self.fc_logvar.forward(x);

        (mu, logvar)
    }
}

// --- DECODER ---

#[derive(Module, Debug)]
pub struct DenseDecoder<B: Backend> {
    layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    activation: Relu,
}

impl<B: Backend> DenseDecoder<B> {
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut current_dim = config.latent_dim;

        // Iterate in REVERSE order to build a symmetric decoder
        // If Encoder is 784 -> 512 -> 256 -> Latent
        // Decoder is Latent -> 256 -> 512 -> 784
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

    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = z;

        // Pass through dynamic hidden layers
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Final layer + Sigmoid for reconstruction
        let x = self.output_layer.forward(x);
        activation::sigmoid(x)
    }
}

// --- FULL VAE ---

#[derive(Module, Debug)]
pub struct DenseVAE<B: Backend> {
    pub encoder: DenseEncoder<B>,
    pub decoder: DenseDecoder<B>,
}

impl<B: Backend> DenseVAE<B> {
    pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
        Self {
            encoder: DenseEncoder::new(config, device),
            decoder: DenseDecoder::new(config, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mu, logvar) = self.encoder.forward(x);
        let z = reparameterize(mu.clone(), logvar.clone());
        let recon_x = self.decoder.forward(z);
        (recon_x, mu, logvar)
    }
}

// use crate::reparameterize;
// use burn::{
//     config::Config,
//     module::Module,
//     nn::{Linear, LinearConfig, Relu},
//     tensor::{Tensor, backend::Backend},
// };

// // --- CONFIG & MODEL DEFINITIONS ---

// /// Configuration for a fully-connected (dense) Variational Autoencoder (VAE).
// ///
// /// This defines the dataset dimensions, architecture hyperparameters,
// /// and training settings. The `Config` derive enables easy serialization
// /// and defaulting via Burn's configuration system.
// #[derive(Config, Debug)]
// pub struct DenseVaeConfig {
//     /// Dimensionality of the input data (e.g. 784 for 28×28 MNIST).
//     #[config(default = 784)]
//     pub input_dim: usize,
//     /// Size of the hidden layer shared across encoder and decoder.
//     #[config(default = 1000)]
//     pub hidden_dim: usize,
//     /// Dimensionality of the latent space used for reparameterization.
//     #[config(default = 20)]
//     pub latent_dim: usize,
//     /// Learning rate to be used by the optimizer.
//     #[config(default = 1e-3)]
//     pub learning_rate: f64,
//     /// Number of epochs to train the model for.
//     #[config(default = 25)]
//     pub num_epochs: usize,
//     /// Batch size used during training.
//     #[config(default = 128)]
//     pub batch_size: usize,
// }

// /// Encoder network for a dense VAE.
// ///
// /// Maps input data → hidden representation → latent mean & log-variance.
// /// Produces `(mu, logvar)`, which are later used for the reparameterization
// /// trick. `Relu` activation is used after the first linear layer.
// #[derive(Module, Debug)]
// pub struct DenseEncoder<B: Backend> {
//     fc1: Linear<B>,
//     fc_mu: Linear<B>,
//     fc_logvar: Linear<B>,
//     activation: Relu,
// }

// /// Decoder network for a dense VAE.
// ///
// /// Maps latent vectors → hidden representation → reconstructed input.
// /// Applies a final sigmoid activation so outputs are in (0,1), suitable
// /// for normalized or binary input data (e.g. MNIST).
// #[derive(Module, Debug)]
// pub struct DenseDecoder<B: Backend> {
//     fc1: Linear<B>,
//     fc2: Linear<B>,
//     activation: Relu,
// }

// impl<B: Backend> DenseDecoder<B> {
//     /// Constructs a new decoder according to the given VAE config.
//     ///
//     /// # Arguments
//     /// * `config` – VAE layer-size configuration.
//     /// * `device` – Backend device to place parameters on.
//     pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
//         Self {
//             fc1: LinearConfig::new(config.latent_dim, config.hidden_dim).init(device),
//             fc2: LinearConfig::new(config.hidden_dim, config.input_dim).init(device),
//             activation: Relu::new(),
//         }
//     }

//     /// Forward pass through the decoder.
//     ///
//     /// # Arguments
//     /// * `z` – Latent sample of shape `(batch_size, latent_dim)`.
//     ///
//     /// # Returns
//     /// Reconstructed input tensor with shape `(batch_size, input_dim)`.
//     pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
//         // Hidden layer
//         let h = self.activation.forward(self.fc1.forward(z));
//         // Output reconstruction with sigmoid activation
//         burn::tensor::activation::sigmoid(self.fc2.forward(h))
//     }
// }

// impl<B: Backend> DenseEncoder<B> {
//     /// Constructs a new encoder according to the given VAE config.
//     ///
//     /// # Arguments
//     /// * `config` – VAE layer-size configuration.
//     /// * `device` – Backend device to place parameters on.
//     pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
//         Self {
//             fc1: LinearConfig::new(config.input_dim, config.hidden_dim).init(device),
//             fc_mu: LinearConfig::new(config.hidden_dim, config.latent_dim).init(device),
//             fc_logvar: LinearConfig::new(config.hidden_dim, config.latent_dim).init(device),
//             activation: Relu::new(),
//         }
//     }

//     /// Forward pass through the encoder.
//     ///
//     /// # Arguments
//     /// * `x` – Input batch with shape `(batch_size, input_dim)`.
//     ///
//     /// # Returns
//     /// A tuple `(mu, logvar)` where:
//     /// * `mu` – Mean of the latent distribution.
//     /// * `logvar` – Log-variance of the latent distribution.
//     pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
//         // Shared hidden representation
//         let h = self.activation.forward(self.fc1.forward(x));
//         // Latent mean
//         let mu = self.fc_mu.forward(h.clone());
//         // Latent log-variance (sigma = logvar)
//         let sigma = self.fc_logvar.forward(h);
//         (mu, sigma)
//     }
// }

// /// Full dense VAE model composed of encoder and decoder.
// ///
// /// Provides methods for sampling via the reparameterization trick
// /// and performing a full forward pass returning reconstructions as
// /// well as latent parameters.
// #[derive(Module, Debug)]
// pub struct DenseVAE<B: Backend> {
//     /// Encoder mapping inputs → latent distribution parameters.
//     pub encoder: DenseEncoder<B>,
//     /// Decoder mapping latent samples → reconstructed inputs.
//     pub decoder: DenseDecoder<B>,
// }

// impl<B: Backend> DenseVAE<B> {
//     /// Constructs a full VAE model from the given configuration.
//     pub fn new(config: &DenseVaeConfig, device: &B::Device) -> Self {
//         Self {
//             encoder: DenseEncoder::new(config, device),
//             decoder: DenseDecoder::new(config, device),
//         }
//     }

//     /// Full VAE forward pass.
//     ///
//     /// # Arguments
//     /// * `x` – Input batch.
//     ///
//     /// # Returns
//     /// A tuple `(recon_x, mu, logvar)` containing:
//     /// * `recon_x` – Reconstruction of the input.
//     /// * `mu` – Latent mean from the encoder.
//     /// * `logvar` – Latent log-variance from the encoder.
//     pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
//         let (mu, sigma) = self.encoder.forward(x);
//         // Sample latent vector via reparameterization
//         let z = reparameterize(mu.clone(), sigma.clone());
//         // Decode the latent sample
//         let recon_x = self.decoder.forward(z);
//         (recon_x, mu, sigma)
//     }
// }
