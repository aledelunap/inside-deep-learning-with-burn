use burn::data::dataloader::batcher::Batcher;
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use inside_deep_learning_with_burn::toy_data::data::{make_toydata, ToyBatcher};
use plotly::{common::Mode, Plot, Scatter};

use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let mut plot = Plot::new();

    let data = make_toydata(0.0, 20.0, 500);

    let x: Vec<f32> = data.iter().map(|item| item.x).collect();
    let y: Vec<f32> = data.iter().map(|item| item.y).collect();

    let trace = Scatter::new(x.clone(), y).mode(Mode::Markers);
    plot.add_trace(trace);

    let batcher = ToyBatcher::<B>::new(device.clone());
    let batch = batcher.batch(data);

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);
    let y = model.forward(batch.x.clone());

    let y = y.flatten::<1>(0, 1).to_data();
    let y = y.convert::<f32>().value;

    let trace = Scatter::new(x, y).mode(Mode::Markers);
    plot.add_trace(trace);

    plot.use_local_plotly();
    plot.write_html(format!("{artifact_dir}/model.html"));
}
