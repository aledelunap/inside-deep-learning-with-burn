mod inference;
mod model;
mod training;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::AdamConfig;

fn main() {
    type ToyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type ToyAutodiffBackend = Autodiff<ToyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "examples/2-1-neural-networks-as-optimization/toy_artifacts";
    training::train::<ToyAutodiffBackend>(
        artifact_dir,
        training::TrainingConfig::new(ModelConfig::new(1, 1), AdamConfig::new()),
        device.clone(),
    );

    inference::infer::<ToyAutodiffBackend>(artifact_dir, device);
}
