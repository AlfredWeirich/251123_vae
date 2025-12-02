use burn::module::ModuleVisitor;
use burn::module::Param;
use burn::tensor::Distribution;
use burn::tensor::{Tensor, backend::Backend};

#[cfg(feature = "conv")]
pub mod conv_model;
#[cfg(feature = "dense")]
pub mod dense_model;
pub mod mnist_data;

/// Width of MNIST images in pixels (28).
pub const MNIST_DIM_X: u32 = 28;

/// Height of MNIST images in pixels (28).
pub const MNIST_DIM_Y: u32 = 28;

/// Dimensionality of the VAE latent space.
///
/// This value is used globally by the inference tool, training process,
/// and CLI interface. Adjusting this changes the representational capacity
/// of the bottleneck layer.
pub const LATENT_DIM: usize = 4;

// --- Loss Function ---

/// Computes the Variational Autoencoder (VAE) loss function.
///
/// This function implements the standard **Evidence Lower Bound (ELBO)** loss,
/// composed of:
///
/// 1. **Reconstruction Loss**  
///    Using Binary Cross-Entropy (BCE):
///    BCE = -[ x*log(x̂) + (1−x)*log(1−x̂) ]
///    This encourages the decoder to accurately reproduce the input.
///
/// 2. **KL Divergence**  
///    Between the approximate posterior `q(z|x)` and the standard normal prior:
///    KL = -0.5 * Σ [ 1 + logσ² − μ² − exp(logσ²) ]
///
/// The final loss is:
/// Loss = BCE + KL
///
/// # Arguments
///
/// * `recon_x` — Decoder output with shape `[batch, input_dim]`  
/// * `x` — Ground-truth inputs, same shape as `recon_x`  
/// * `mu` — Latent mean from encoder, shape `[batch, latent_dim]`  
/// * `sigma` — Latent log-variance from encoder, shape `[batch, latent_dim]`  
///
/// # Returns
///
/// A tensor containing a single scalar loss value (shape `[1]`), averaged across the batch.
pub fn loss_function<B: Backend>(
    recon_x: Tensor<B, 2>,
    x: Tensor<B, 2>,
    mu: Tensor<B, 2>,
    sigma: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // Numerical stability for log(x) calls
    let eps = 1e-8;

    // Clamp predicted pixels so log() never receives 0.0 or 1.0
    let recon_clamp = recon_x.clamp(eps, 1.0 - eps);

    // Binary cross-entropy:
    // BCE = -( x * log(x̂) + (1 - x) * log(1 - x̂) )
    let bce = (x.clone() * recon_clamp.clone().log()
        + x.clone().neg().add_scalar(1.0) * recon_clamp.neg().add_scalar(1.0).log())
    .neg();

    // Reconstruction loss: mean over batch
    let recon_loss = bce.sum_dim(1).mean();

    // KL divergence:
    // KL = 0.5 * Σ( exp(logvar) + mu^2 - logvar - 1 )
    let kld = (sigma.clone().exp() + mu.powf_scalar(2.0) - sigma - 1.0)
        .sum_dim(1)
        .mean()
        .mul_scalar(0.5);

    // Total loss
    recon_loss + kld
}

/// Convert grayscale floating-point pixel values `[0.0–1.0]`
/// into RGB byte values `[0–255]` triplicated (R=G=B).
///
/// This is required because the chart viewer expects raw RGB pixel buffers.
pub fn build_rgb_bytes(pixels: &[f32]) -> Vec<u8> {
    let mut res = Vec::with_capacity(pixels.len() * 3);
    for &val in pixels {
        let u = (val * 255.0).clamp(0.0, 255.0) as u8;
        res.push(u); // R
        res.push(u); // G
        res.push(u); // B
    }
    res
}

/// Applies the reparameterization trick to obtain a latent sample.
/// Shared logic for all VAE models.
pub fn reparameterize<B: Backend, const D: usize>(
    mu: Tensor<B, D>,
    logvar: Tensor<B, D>,
) -> Tensor<B, D> {
    // Convert log-variance to standard deviation
    // std = exp(0.5 * logvar)
    let std = logvar.mul_scalar(0.5).exp();

    // Sample epsilon ~ N(0, 1)
    let eps = Tensor::random_like(&std, Distribution::Normal(0.0, 1.0));

    // z = mu + eps * std
    mu + eps * std
}


/// A visitor that prints the model's module tree and parameters.
pub struct ModelTreePrinter {
    indent: usize,
}

impl ModelTreePrinter {
    pub fn new() -> Self {
        Self { indent: 0 }
    }

    fn print_indent(&self) {
        print!("{}", "  ".repeat(self.indent));
    }
}

// 2. Implement the ModuleVisitor trait
impl<B: Backend> ModuleVisitor<B> for ModelTreePrinter {
    // Called when entering a module (e.g., "conv1", "layer_norm")
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.print_indent();
        println!("Module: {}", name);
        self.indent += 1;
    }

    // Called when exiting a module
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.indent -= 1;
    }

    // Called for every float tensor parameter (weights, biases)
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.print_indent();
        // You can inspect the tensor here (e.g., param.shape())
        println!("Param (Float): {:?} [ID: {}]", param.shape(), param.id);
    }

    // You can also implement visit_int and visit_bool if needed
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, burn::tensor::Int>>) {
        self.print_indent();
        println!("Param (Int): {:?}", param.shape());
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, burn::tensor::Bool>>) {
        self.print_indent();
        println!("Param (Bool): {:?}", param.shape());
    }
}
