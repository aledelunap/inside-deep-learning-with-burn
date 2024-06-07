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

#[derive(Clone, Debug)]
pub struct ToyItem {
    pub x: f32,
    pub y: f32,
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

impl ToyDataset {
    const START: f32 = 0.0;
    const END: f32 = 20.0;
    const N: usize = 5000;
    const SPLIT: f32 = 0.8;

    pub fn train() -> Self {
        Self::new("train")
    }
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let x = Array::<f32, Ix1>::linspace(Self::START, Self::END, Self::N);
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();

        let y = x.mapv(|x| x + x.sin() + normal.sample(&mut rng));
        let toy_items = x
            .into_iter()
            .zip(y.into_iter())
            .map(|(x, y)| ToyItem { x, y });

        let amount = (Self::SPLIT * Self::N as f32).ceil() as usize;

        let items = match split {
            "train" => toy_items.choose_multiple(&mut rng, amount),
            _ => toy_items
                .choose_multiple(&mut rng, Self::N)
                .drain(0..amount)
                .collect(),
        };

        Self { dataset: items }
    }
}
