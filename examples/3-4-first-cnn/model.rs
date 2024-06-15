use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig,
    },
    prelude::*,
};
use nn::PaddingConfig2d;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv: Conv2d<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    image_height: usize,
    image_width: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv: Conv2dConfig::new([1, 16], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            linear: LinearConfig::new(16 * self.image_height * self.image_width, self.num_classes)
                .init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv.forward(x); // [batch_size, 16, width, height]
        let x = x.tanh();
        let x = x.reshape([batch_size, 16 * height * width]);
        self.linear.forward(x)
    }
}
