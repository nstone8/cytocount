use ciborium;
use clap::Parser;
use cytocount::{coords_to_df, debug_images};
use imageproc::map::map_pixels;
use papillae::ralston;
use polars::prelude::*;
use ralston::image::{ImageBuffer, Luma};
use recbudd;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
#[derive(Parser)]
struct MyArgs {
    file_path: PathBuf,
    blur: f32,
    threshold: u8,
    min_area: u64,
    min_frames: usize,
    time_window: f32,
    tolerance: f32,
}

fn main() {
    let args = MyArgs::parse();
    let path = args.file_path.clone();
    let f = File::open(&path).expect("couldn't open file");
    let mut reader = BufReader::new(f);
    let mut dir = PathBuf::new();
    dir.push(path.parent().unwrap());
    dir.push(format!(
        "{}_processed",
        path.file_name().unwrap().to_str().unwrap()
    ));
    //load all of our images into a vector
    let mut frame_vec = Vec::<ImageBuffer<Luma<u8>, Vec<u8>>>::new();
    //modify this loop to load the frames
    println!("reading images");
    loop {
        match ciborium::from_reader::<recbudd::RecFrame, &mut BufReader<File>>(&mut reader) {
            Ok(rec_frame) => {
                let im = rec_frame.to_image().into_luma8();
                frame_vec.push(im);
            }
            Err(_) => {
                break;
            }
        }
    }
    println!("calculating background");
    let bg = map_pixels(&frame_vec[0], |x, y, p| {
        let first_frame_value: u64 = p[0].into();
        let other_frames_sum: u64 = frame_vec[1..]
            .iter()
            .map(|im| -> u64 { im.get_pixel(x, y)[0].into() })
            .sum();
        let vec_len: u64 = frame_vec.len().try_into().unwrap();
        let this_pixel_average: u8 = ((first_frame_value + other_frames_sum) / vec_len)
            .try_into()
            .unwrap();
        [this_pixel_average].into()
    });
    println!("writing images");
    let paths = debug_images(
        args.file_path,
        dir.clone(),
        //find_objects args
        &bg,
        args.blur,
        args.threshold,
        args.min_area,
        //track_paths args
        args.min_frames,
        args.time_window,
        args.tolerance,
    );
    let df_vec: Vec<LazyFrame> = paths
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let path_error = p.max_error();
            let path_v1 = p.v1;
            let path_v2 = p.v2;
            let this_df = coords_to_df(&p.into_vec()).lazy();
            this_df.with_columns([
                lit(i as u32).alias("path_index"),
                lit(path_error).alias("max error"),
                lit(path_v1).alias("v1"),
                lit(path_v2).alias("v2"),
            ])
        })
        .collect();
    let mut df_path = dir.clone();
    df_path.push("paths.csv");
    let df_file = File::create(df_path).unwrap();
    let mut cw = CsvWriter::new(df_file);
    cw.finish(
        &mut concat(&df_vec, Default::default())
            .expect("couldn't concatenate dataframes")
            .collect()
            .unwrap(),
    )
    .unwrap();
}
