use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Data, Tensor},
};
use ndarray::{Array, Ix1};
use ndarray_rand::{
    rand,
    rand_distr::{Distribution, Normal},
};
use plotly::{common::Mode, Plot, Scatter};

use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let mut plot = Plot::new();

    let x = Array::<f32, Ix1>::linspace(0.0, 20.0, 2000);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let y = x.map(|x| x + x.sin() + normal.sample(&mut rand::thread_rng()));

    let trace = Scatter::new(x.to_vec(), y.to_vec()).mode(Mode::Markers);
    plot.add_trace(trace);

    let x = x
        .iter()
        .map(|x| Data::<f32, 1>::from([*x]))
        .map(|data| Tensor::<B, 1>::from_data(data.convert(), &device))
        .map(|tensor| tensor.reshape([1, 1]))
        .collect();
    let x = Tensor::cat(x, 0).to_device(&device);

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);
    let y = model.forward(x.clone());

    let x = x.flatten::<1>(0, 1).to_data();
    let y = y.flatten::<1>(0, 1).to_data();

    let x = x.convert::<f32>().value;
    let y = y.convert::<f32>().value;

    let trace = Scatter::new(x, y).mode(Mode::Markers);
    plot.add_trace(trace);

    plot.use_local_plotly();
    plot.write_html(format!("{artifact_dir}/linear_model.html"));
}
