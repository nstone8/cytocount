use ciborium;
use clap::Parser;
use cytocount::{coords_to_df, find_objects, track_paths, ObjCoords, PathStatus};
use imageproc::map::map_pixels;
use papillae::ralston;
use polars::prelude::{CsvWriter, SerWriter};
use ralston::image::{ImageBuffer, Luma};
use recbudd;
use std::fs::{DirBuilder, File};
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
    let path = args.file_path;
    let f = File::open(&path).expect("couldn't open file");
    let mut reader = BufReader::new(f);
    let mut dir = PathBuf::new();
    dir.push(path.parent().unwrap());
    dir.push(format!(
        "{}_processed",
        path.file_name().unwrap().to_str().unwrap()
    ));
    let b = DirBuilder::new();
    b.create(&dir).expect("couldn't create image directory");
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

    let mut framenum = 1;
    //loop here to process and save the frames
    println!("processing frames");
    let mut oc = Vec::<ObjCoords>::new();
    for im in frame_vec {
        let mut im_path = dir.clone();
        im_path.push(format!("{}.png", framenum));
        framenum += 1;
        let (proc, cent_vec) = find_objects(&bg, &im, args.blur, args.threshold, args.min_area);
        proc.save(im_path).expect("couldn't save image");
        let mut o: Vec<ObjCoords> = cent_vec
            .into_iter()
            .map(|c| ObjCoords {
                x: c.0,
                y: c.1,
                t: framenum as f32,
            })
            .collect();
        oc.append(&mut o);
    }
    let mut df = coords_to_df(&oc);
    let mut df_path = dir.clone();
    df_path.push("centroids.csv");
    let df_file = File::create(df_path).unwrap();
    let mut cw = CsvWriter::new(df_file);
    cw.finish(&mut df).unwrap();

    let mut objs_pathstatus: Vec<PathStatus> =
        oc.into_iter().map(|o| PathStatus::OffPath(o)).collect();
    let _ = track_paths(
        &mut objs_pathstatus,
        args.min_frames,
        args.time_window,
        args.tolerance,
    );
}
