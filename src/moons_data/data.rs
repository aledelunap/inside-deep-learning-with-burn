use burn::data::dataset::Dataset;
use ndarray::{concatenate, stack, Array, Axis, Ix1, Ix2};
use ndarray_rand::rand::rngs;
use ndarray_rand::rand::seq::{IteratorRandom, SliceRandom};
use ndarray_rand::rand::SeedableRng;
use std::f32::consts::PI;

pub fn make_moons(n_inner: usize, n_outer: usize) -> Vec<MoonsItem> {
    let inner_moon_x = Array::<f32, Ix1>::linspace(0.0, PI, n_inner).map(|x| 1.0 - x.cos());
    let inner_moon_y = Array::<f32, Ix1>::linspace(0.0, PI, n_inner).map(|x| 1.0 - x.sin() - 0.5);

    let inner_moon = stack(Axis(1), &[inner_moon_x.view(), inner_moon_y.view()]).unwrap();
    let inner_y = Array::<i8, Ix1>::ones(n_inner);

    let outer_moon_x = Array::<f32, Ix1>::linspace(0.0, PI, n_outer).map(|x| x.cos());
    let outer_moon_y = Array::<f32, Ix1>::linspace(0.0, PI, n_outer).map(|x| x.sin());

    let outer_moon: Array<f32, Ix2> =
        stack(Axis(1), &[outer_moon_x.view(), outer_moon_y.view()]).unwrap();
    let outer_y = Array::<i8, Ix1>::zeros(n_outer);

    let x: Array<f32, Ix2> = concatenate(Axis(0), &[inner_moon.view(), outer_moon.view()]).unwrap();
    let y: Array<i8, Ix1> = concatenate(Axis(0), &[inner_y.view(), outer_y.view()]).unwrap();

    let dataset = x
        .axis_iter(Axis(0))
        .zip(y.into_iter())
        .map(|(x, y)| {
            let x = match x.to_vec()[..] {
                [x1, x2] => [x1, x2],
                _ => {
                    panic!("Expected two values");
                }
            };
            MoonsItem { x, y }
        })
        .collect();

    dataset
}

#[derive(Clone, Debug)]
pub struct MoonsItem {
    pub x: [f32; 2],
    pub y: i8,
}

pub struct MoonsDataset {
    dataset: Vec<MoonsItem>,
}

impl Dataset<MoonsItem> for MoonsDataset {
    fn get(&self, index: usize) -> Option<MoonsItem> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MoonsDataset {
    const N_INNER: usize = 2000;
    const N_OUTER: usize = 2000;
    const SPLIT: f32 = 0.8;

    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let mut rng = rngs::StdRng::seed_from_u64(12);
        let mut items = make_moons(Self::N_INNER, Self::N_OUTER);
        items.shuffle(&mut rng);
        let amount = (Self::SPLIT * (Self::N_INNER + Self::N_OUTER) as f32).ceil() as usize;

        let split_items = match split {
            "train" => items.into_iter().choose_multiple(&mut rng, amount),
            _ => items
                .into_iter()
                .choose_multiple(&mut rng, Self::N_INNER + Self::N_OUTER)
                .drain(0..amount)
                .collect(),
        };

        Self {
            dataset: split_items,
        }
    }
}
