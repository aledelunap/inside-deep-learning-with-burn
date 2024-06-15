mod data;
mod model;
mod training;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::AdamConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        "examples/3-4-first-cnn/artifacts/",
        crate::training::TrainingConfig::new(ModelConfig::new(10, 28, 28), AdamConfig::new()),
        device,
    );
}
