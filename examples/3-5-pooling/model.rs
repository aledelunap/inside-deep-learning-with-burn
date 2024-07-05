use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig,
    },
    prelude::*,
};
use nn::{
    pool::{MaxPool2d, MaxPool2dConfig},
    PaddingConfig2d,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool1: MaxPool2d,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    pool2: MaxPool2d,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    filters: usize,
    image_height: usize,
    image_width: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([self.filters, self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([self.filters, self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv4: Conv2dConfig::new([self.filters, 2 * self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv5: Conv2dConfig::new([2 * self.filters, 2 * self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv6: Conv2dConfig::new([2 * self.filters, 2 * self.filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            linear: LinearConfig::new(
                2 * self.filters * (self.image_height / 4) * (self.image_width / 4),
                self.num_classes,
            )
            .init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 16, width, height]
        let x = x.tanh();
        let x = self.conv2.forward(x); // [batch_size, 16, width, height]
        let x = x.tanh();
        let x = self.conv3.forward(x); // [batch_size, 16, width, height]
        let x = x.tanh();
        let x = self.pool1.forward(x); // [batch_size, 16, width / 2, height / 2]
        let x = self.conv4.forward(x); // [batch_size, 2 * 16, width / 2, height / 2]
        let x = x.tanh();
        let x = self.conv5.forward(x); // [batch_size, 2 * 16, width / 2, height / 2]
        let x = x.tanh();
        let x = self.conv6.forward(x); // [batch_size, 2 * 16, width / 2, height / 2]
        let x = x.tanh();
        let x = self.pool2.forward(x); // [batch_size, 2 * 16, width / 4, height / 4]
        let x = x.reshape([batch_size, 2 * 16 * (height / 4) * (width / 4)]);
        self.linear.forward(x)
    }
}
