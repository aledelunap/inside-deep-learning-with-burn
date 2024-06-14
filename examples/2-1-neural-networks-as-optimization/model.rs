use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    in_features: usize,
    out_features: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear: LinearConfig::new(self.in_features, self.out_features).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    // Shapes
    // - x: [batch_size, in_features]
    // - y: [batch_size, out_features]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}
