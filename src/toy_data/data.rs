use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::{Float, Tensor};
use burn::tensor::backend::Backend;
use burn::tensor::Data;
use ndarray::{Array, Ix1};
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{
    rand,
    rand_distr::{Distribution, Normal},
};

pub fn make_toydata(start: f32, end: f32, n: usize) -> Vec<ToyItem> {
    let x = Array::<f32, Ix1>::linspace(start, end, n);

    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(21);
    let y = x.mapv(|x| x + x.sin() + normal.sample(&mut rng));

    let toy_items = x
        .into_iter()
        .zip(y.into_iter())
        .map(|(x, y)| ToyItem { x, y })
        .collect();

    toy_items
}

#[derive(Clone, Debug)]
pub struct ToyItem {
    pub x: f32,
    pub y: f32,
}

pub struct ToyDataset {
    dataset: Vec<ToyItem>,
}

impl Dataset<ToyItem> for ToyDataset {
    fn get(&self, index: usize) -> Option<ToyItem> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub struct ToyDatasetConfig {
    pub start: f32,
    pub end: f32,
    pub n: usize,
    pub split: f32,
}

impl ToyDatasetConfig {
    pub fn train(&self) -> ToyDataset {
        self.new("train")
    }
    pub fn test(&self) -> ToyDataset {
        self.new("test")
    }

    fn new(&self, split: &str) -> ToyDataset {
        let toy_items = make_toydata(self.start, self.end, self.n);

        let amount = (self.split * self.n as f32).ceil() as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);

        let items = match split {
            "train" => toy_items.into_iter().choose_multiple(&mut rng, amount),
            _ => toy_items
                .into_iter()
                .choose_multiple(&mut rng, self.n)
                .drain(0..amount)
                .collect(),
        };

        ToyDataset { dataset: items }
    }
}

#[derive(Clone)]
pub struct ToyBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ToyBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct ToyBatch<B: Backend> {
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 2>,
}

impl<B: Backend> Batcher<ToyItem, ToyBatch<B>> for ToyBatcher<B> {
    fn batch(&self, items: Vec<ToyItem>) -> ToyBatch<B> {
        let x = items
            .iter()
            .map(|item| Data::<f32, 1>::from([item.x]))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let y = items
            .iter()
            .map(|item| item.y)
            .map(|y| Data::<f32, 1>::from([y]))
            .map(|data| Tensor::<B, 1, Float>::from_data(data.convert(), &self.device))
            .map(|tensor: Tensor<B, 1, Float>| tensor.reshape([1, 1]))
            .collect();

        let x = Tensor::cat(x, 0).to_device(&self.device);
        let y = Tensor::cat(y, 0).to_device(&self.device);

        ToyBatch { x, y }
    }
}
