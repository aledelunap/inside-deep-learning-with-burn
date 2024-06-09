use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Int, Tensor},
};

use super::data::MoonsItem;

#[derive(Clone)]
pub struct MoonsBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MoonsBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MoonsBatch<B: Backend> {
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MoonsItem, MoonsBatch<B>> for MoonsBatcher<B> {
    fn batch(&self, items: Vec<MoonsItem>) -> MoonsBatch<B> {
        let x = items
            .iter()
            .map(|item| Data::<f32, 2>::from([item.x]))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .collect();

        let y = items
            .iter()
            .map(|item| item.y)
            .map(|y| Data::<i8, 1>::from([y]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device))
            .collect();

        let x = Tensor::cat(x, 0).to_device(&self.device);
        let y = Tensor::cat(y, 0).to_device(&self.device);

        MoonsBatch { x, y }
    }
}
