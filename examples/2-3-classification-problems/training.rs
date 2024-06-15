use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Float, Int, Tensor,
    },
    train::{
        metric::AccuracyMetric, ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep,
        ValidStep,
    },
};

use inside_deep_learning_with_burn::moons_data::{self, data::MoonDatasetConfig};
use moons_data::batcher::{MoonsBatch, MoonsBatcher};

use crate::model::{Model, ModelConfig};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        x: Tensor<B, 2, Float>,
        y: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(x);
        let loss = CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), y.clone());

        ClassificationOutput::new(loss, output, y)
    }
}

impl<B: AutodiffBackend> TrainStep<MoonsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MoonsBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.x, batch.y);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MoonsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MoonsBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.x, batch.y)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 250)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 32)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = MoonsBatcher::<B>::new(device.clone());
    let batcher_test = MoonsBatcher::<B::InnerBackend>::new(device.clone());

    let data = MoonDatasetConfig {
        n_inner: 500,
        n_outer: 500,
        split: 0.9,
        noise: 0.01,
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(data.train());

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(data.test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let trained_model = learner.fit(dataloader_train, dataloader_test);

    trained_model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
