use std::cmp::Ordering;

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::ElementComparison;
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use inside_deep_learning_with_burn::moons_data::{batcher::MoonsBatcher, data::make_moons};

use plotly::color::NamedColor;
use plotly::common::Marker;
use plotly::{common::Mode, Plot, Scatter};

use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let mut plot = Plot::new();

    let data = make_moons(100, 100);

    let x1: Vec<f32> = data.iter().map(|item| item.x[0]).collect();
    let x2: Vec<f32> = data.iter().map(|item| item.x[1]).collect();
    let y: Vec<NamedColor> = data
        .iter()
        .map(|item| match item.y {
            0 => NamedColor::Red,
            _ => NamedColor::Blue,
        })
        .collect();

    let trace = Scatter::new(x1.clone(), x2.clone())
        .name("True class")
        .mode(Mode::Markers)
        .marker(Marker::new().color_array(y));

    plot.add_trace(trace);

    let batcher = MoonsBatcher::<B>::new(device.clone());
    let batch = batcher.batch(data);

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);
    let y = model.forward(batch.x.clone());
    let y = y
        .argmax(1)
        .flatten::<1>(0, 1)
        .to_data()
        .convert::<i8>()
        .value
        .iter()
        .map(|y| match y {
            1 => NamedColor::Red,
            _ => NamedColor::Blue,
        })
        .collect();

    let trace = Scatter::new(x1, x2)
        .name("Predicted class")
        .mode(Mode::Markers)
        .marker(Marker::new().color_array(y));

    plot.add_trace(trace);

    plot.use_local_plotly();
    plot.write_html(format!("{artifact_dir}/model.html"));
}
