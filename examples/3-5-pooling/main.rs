mod inference;
mod model;
mod training;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;

fn main() {
    let artifact_dir = "examples/3-5-pooling/artifacts/";

    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        crate::training::TrainingConfig::new(ModelConfig::new(10, 16, 28, 28), AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
