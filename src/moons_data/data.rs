use burn::data::dataset::Dataset;
use ndarray::{concatenate, stack, Array, Axis, Ix1, Ix2};
use ndarray_rand::rand::seq::{IteratorRandom, SliceRandom};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::{self, rngs};
use ndarray_rand::rand_distr::{Distribution, Normal};
use std::f32::consts::PI;

pub fn make_moons(n_inner: usize, n_outer: usize, noise: f32) -> Vec<MoonsItem> {
    let normal = Normal::<f32>::new(0.0, noise).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(21);

    let inner_moon_x = Array::<f32, Ix1>::linspace(0.0, PI, n_inner)
        .map(|x| 1.0 - x.cos() + normal.sample(&mut rng));
    let inner_moon_y = Array::<f32, Ix1>::linspace(0.0, PI, n_inner)
        .map(|x| 1.0 - x.sin() - 0.5 + normal.sample(&mut rng));

    let inner_moon = stack(Axis(1), &[inner_moon_x.view(), inner_moon_y.view()]).unwrap();
    let inner_class = Array::<i8, Ix1>::ones(n_inner);

    let outer_moon_x =
        Array::<f32, Ix1>::linspace(0.0, PI, n_outer).map(|x| x.cos() + normal.sample(&mut rng));
    let outer_moon_y =
        Array::<f32, Ix1>::linspace(0.0, PI, n_outer).map(|x| x.sin() + normal.sample(&mut rng));

    let outer_moon: Array<f32, Ix2> =
        stack(Axis(1), &[outer_moon_x.view(), outer_moon_y.view()]).unwrap();
    let outer_class = Array::<i8, Ix1>::zeros(n_outer);

    let x: Array<f32, Ix2> = concatenate(Axis(0), &[inner_moon.view(), outer_moon.view()]).unwrap();
    let y: Array<i8, Ix1> =
        concatenate(Axis(0), &[inner_class.view(), outer_class.view()]).unwrap();

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

pub struct MoonDatasetConfig {
    pub n_inner: usize,
    pub n_outer: usize,
    pub split: f32,
    pub noise: f32,
}

impl MoonDatasetConfig {
    pub fn train(&self) -> MoonsDataset {
        self.new("train")
    }

    pub fn test(&self) -> MoonsDataset {
        self.new("test")
    }

    fn new(&self, split: &str) -> MoonsDataset {
        let mut rng = rngs::StdRng::seed_from_u64(12);
        let mut items = make_moons(self.n_inner, self.n_outer, self.noise);
        items.shuffle(&mut rng);
        let amount = (self.split * (self.n_inner + self.n_outer) as f32).ceil() as usize;

        let split_items = match split {
            "train" => items.into_iter().choose_multiple(&mut rng, amount),
            _ => items
                .into_iter()
                .choose_multiple(&mut rng, self.n_inner + self.n_outer)
                .drain(0..amount)
                .collect(),
        };

        MoonsDataset {
            dataset: split_items,
        }
    }
}
