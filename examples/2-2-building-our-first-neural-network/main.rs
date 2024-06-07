use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    optim::AdamConfig,
};

mod inference;
mod model;
mod training;

fn main() {
    type ToyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type ToyAutodiffBackend = Autodiff<ToyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "examples/2-2-building-our-first-neural-network/toy_artifacts";
    training::train::<ToyAutodiffBackend>(
        artifact_dir,
        training::TrainingConfig::new(model::ModelConfig::new(1, 10, 1), AdamConfig::new())
            .with_num_epochs(256),
        device.clone(),
    );

    inference::infer::<ToyAutodiffBackend>(artifact_dir, device);
}
