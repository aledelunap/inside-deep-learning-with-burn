use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Float, Tensor,
    },
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::model::{Model, ModelConfig};
use inside_deep_learning_with_burn::toy_data;
use toy_data::data::{ToyBatch, ToyBatcher, ToyDatasetConfig};

impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        x: Tensor<B, 2, Float>,
        y: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output = self.forward(x);
        let loss = MseLoss::new().forward(output.clone(), y.clone(), Mean);

        RegressionOutput::new(loss, output, y)
    }
}

impl<B: AutodiffBackend> TrainStep<ToyBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: ToyBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.x, batch.y);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ToyBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: ToyBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.x, batch.y)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 32)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 32)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-2)]
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

    let batcher_train = ToyBatcher::<B>::new(device.clone());
    let batcher_test = ToyBatcher::<B::InnerBackend>::new(device.clone());

    let data = ToyDatasetConfig {
        start: 0.0,
        end: 20.0,
        n: 1000,
        split: 0.9,
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
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
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
