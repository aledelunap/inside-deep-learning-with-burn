mod inference;
mod model;
mod training;

use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    optim::AdamConfig,
};

fn main() {
    type MoonsBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MoonsAutodiffBackend = Autodiff<MoonsBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "examples/2-3-classification-problems/artifacts";

    training::train::<MoonsAutodiffBackend>(
        artifact_dir,
        training::TrainingConfig::new(model::ModelConfig::new(2, 30, 2), AdamConfig::new()),
        device.clone(),
    );

    inference::infer::<MoonsBackend>(artifact_dir, device);
}
