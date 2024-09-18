use image::{imageops::overlay, DynamicImage, GrayImage, ImageBuffer, Pixel, RgbaImage};
use imageproc::region_labelling::{connected_components, Connectivity};
use imageproc::{
    contrast::{threshold, ThresholdType},
    definitions::Image,
    filter::gaussian_blur_f32,
    image,
    map::map_pixels,
};
use itertools::Itertools;
use moore_penrose::{pinv, Dim, Dyn, OMatrix};
use polars::prelude::{df, DataFrame};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{DirBuilder, File};
use std::io::{BufReader, Seek, SeekFrom};
use std::ops::Deref;
use std::path::PathBuf;

///helper function for making debug images
pub fn hcat_image<P, C>(v: &[&ImageBuffer<P, C>]) -> Image<P>
where
    P: Pixel,
    C: Deref<Target = [P::Subpixel]>,
{
    let new_h = v.iter().map(|i| i.height()).max().unwrap();
    let new_w = v.iter().map(|i| i.width()).sum();
    let mut new_im = ImageBuffer::<P, Vec<P::Subpixel>>::new(new_w, new_h);
    let mut cur_x: u32 = 0;
    for im in v {
        overlay(&mut new_im, *im, cur_x.into(), 0);
        cur_x += im.width();
    }
    return new_im;
}

///subtract one image from another (useful for removing background).
///`image_diff(a,b)` performs `abs(a-b)`
fn image_diff(a: &GrayImage, b: &GrayImage) -> GrayImage {
    map_pixels(a, |x, y, p| {
        //indexing gets us the primitive (numeric value).
        let raw_value: i32 = (p[0] as i32) - (b.get_pixel(x, y)[0] as i32);
        //println!("{} - {} = {}",p[0],b.get_pixel(x,y)[0],raw_value);
        let abs_value: u8 = raw_value.abs().try_into().unwrap();
        //a one length array of T implements Into<Luma<T>>
        [abs_value].into()
    })
}

/*
///"invert" image (make light areas dark and vice versa)
fn inv_image(im:&GrayImage) -> GrayImage{
    map_pixels(im, |_,_,p| {
    [255 - p[0]].into()
    })
}
*/

///Little helper struct for calculating object size and centroid
struct DetectedObject {
    num_pix: u64,
    sum_x: u64,
    sum_y: u64,
}

impl DetectedObject {
    fn new() -> Self {
        DetectedObject {
            num_pix: 0,
            sum_x: 0,
            sum_y: 0,
        }
    }
    fn add_pixel(&mut self, x: u32, y: u32) {
        self.num_pix += 1;
        self.sum_x += x as u64;
        self.sum_y += y as u64;
    }
    fn get_centroid(&self) -> (u32, u32) {
        (
            (self.sum_x / self.num_pix).try_into().unwrap(),
            (self.sum_y / self.num_pix).try_into().unwrap(),
        )
    }
}

/*
///Draw a 10x10 red box
fn red_box() -> RgbImage {
    rgb_image!(
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0];
    [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]
    )
}
*/

///Draw a line with `line_weight` from pixel `a` to `b` filled with rgb color `c` on a transparent
///background with `width` and `height`
fn line_overlay(
    width: u32,
    height: u32,
    a: [u32; 2],
    b: [u32; 2],
    line_weight: u32,
    c: [u8; 3],
) -> RgbaImage {
    //vector from a to b
    let v: Vec<f64> = a
        .into_iter()
        .zip(b.into_iter())
        .map(|coords| {
            let c0: f64 = coords.0.into();
            let c1: f64 = coords.1.into();
            c0 - c1
        })
        .collect();
    //vector normal to v
    let n = [-v[1], v[0]];
    //magnitude of n
    let mag_n: f64 = (n[0].powi(2) + n[1].powi(2)).sqrt();
    //little helper closure
    let point_on_line = |x: f64, y: f64| {
        //vector from a to our point
        let v_p = [x - (a[0] as f64), y - (a[1] as f64)];
        //distance from our line to (x,y) is abs((v_p⋅n)/mag_n)
        let dot_prod: f64 = v_p[0] * n[0] + v_p[1] * n[1];
        let dist = (dot_prod / mag_n).abs();
        if dist > line_weight.into() {
            false
        } else {
            //only want to plot the line between the points, not infinite length
            let mut x_sorted = [a[0], b[0]];
            x_sorted.sort();
            let mut y_sorted = [a[1], b[1]];
            y_sorted.sort();
            let line_weight_f = line_weight as f64;
            //convert x_sorted and y_sorted to f64
            let x_sorted_f: Vec<_> = x_sorted.into_iter().map(|k| k as f64).collect();
            let y_sorted_f: Vec<_> = y_sorted.into_iter().map(|k| k as f64).collect();
            //is this pixel within a bounding box set by the points
            (x_sorted_f[0] - line_weight_f < x)
                && (x < x_sorted_f[1] + line_weight_f)
                && (y_sorted_f[0] - line_weight_f < y)
                && (y < y_sorted_f[1] + line_weight_f)
        }
    };

    RgbaImage::from_fn(width, height, |x, y| {
        if point_on_line(x.into(), y.into()) {
            //return an opaque pixel with color c
            [c[0], c[1], c[2], 255].into()
        } else {
            //return a transparent pixel
            [0, 0, 0, 0].into()
        }
    })
}

///Draw a circle at 'center' with radius `r` filled with rgb color `c` on a transparent background
///with `width` and `height`
fn circle_overlay(width: u32, height: u32, r: u32, center: [u32; 2], c: [u8; 3]) -> RgbaImage {
    let point_in_circle = |x: f64, y: f64| {
        let v: [f64; 2] = [x - (center[0] as f64), y - (center[1] as f64)];
        let dist = (v[0].powi(2) + v[1].powi(2)).sqrt();
        dist <= r.into()
    };

    RgbaImage::from_fn(width, height, |x, y| {
        if point_in_circle(x.into(), y.into()) {
            //return an opaque pixel with color c
            [c[0], c[1], c[2], 255].into()
        } else {
            //return a transparent pixel
            [0, 0, 0, 0].into()
        }
    })
}

///Detect objects. This function returns a tuple where the the first entry is an `Option`al image
///showing the various phases of the image processing and the second entry is a `Vec` of object
///centroids.
pub fn find_objects(
    bg: &GrayImage,
    f: &GrayImage,
    blur: f32,
    threshold_value: u8,
    min_obj_area: u64,
    build_debug_image: bool,
) -> (Option<RgbaImage>, Vec<(u32, u32)>) {
    //invert our inputs
    //let bg = inv_image(bg);
    //let f = inv_image(f);
    //subtract off the background
    let diff = image_diff(&f, &bg);
    //first do a blur to reduce noise
    let blurred = gaussian_blur_f32(&diff, blur);
    //now threshold
    let thresh = threshold(&blurred, threshold_value, ThresholdType::Binary);
    //label the connected objects
    let conn = connected_components(&thresh, Connectivity::Eight, [0].into());
    //filter small objects
    let mut objs = HashMap::<u32, DetectedObject>::new();
    for x in 0..conn.width() {
        for y in 0..conn.height() {
            let p = conn.get_pixel(x, y)[0];
            if p != 0 {
                //if p == 0 this is a background pixel
                match objs.get_mut(&p) {
                    Some(o) => {
                        o.add_pixel(x, y);
                    }
                    None => {
                        let mut o = DetectedObject::new();
                        o.add_pixel(x, y);
                        objs.insert(p, o);
                    }
                }
            }
        }
    }
    //extract coordinates of object centroids
    let mut centroids = Vec::<_>::new();
    for (_, v) in objs.iter() {
        if v.num_pix >= min_obj_area.into() {
            centroids.push(v.get_centroid())
        }
    }
    //println!("centroid list: {:?}", centroids);
    let output_im = if build_debug_image {
        let debug_im =
            DynamicImage::ImageLuma8(hcat_image(&[&f, &bg, &diff, &blurred, &thresh])).into_rgba8();
        let mut labeled_im = DynamicImage::ImageLuma8(f.clone()).into_rgba8();
        let im_h = labeled_im.height();
        let im_w = labeled_im.width();
        for c in centroids.iter() {
            overlay(
                &mut labeled_im,
                &circle_overlay(im_w, im_h, 10, [c.0, c.1], [255, 0, 0]),
                0,
                0,
            );
        }
        Some(hcat_image(&[&debug_im, &labeled_im]))
    } else {
        None
    };
    return (output_im, centroids);
}

///Struct representing the coordinates of a detected object
#[derive(Debug, Clone)]
pub struct ObjCoords {
    pub x: u32,
    pub y: u32,
    pub t: f32,
}

///Convert a list of [ObjCoords] into a [DataFrame]
pub fn coords_to_df(oc: &[ObjCoords]) -> DataFrame {
    let x: Vec<_> = oc.iter().map(|o| o.x).collect();
    let y: Vec<_> = oc.iter().map(|o| o.y).collect();
    let t: Vec<_> = oc.iter().map(|o| o.t).collect();
    df!("x" => x, "y" => y, "t" => t).unwrap()
}

///Struct representing the path taken by an Object
#[derive(Debug)]
pub struct ObjPath {
    path: Vec<ObjCoords>,
    err: f64,
    pub v1: f64,
    pub v2: f64,
}

impl ObjPath {
    /*
    ///Create a new `ObjPath`
    pub fn new() -> Self {
        ObjPath {
            path: Vec::<ObjCoords>::new(),
        }
    }

    ///Add a point to this path
    pub fn push(&mut self, point: ObjCoords) {
        self.path.push(point);
    }
     */
    ///Get the max error associated with this path
    pub fn max_error(&self) -> f64 {
        self.err
    }
    ///Consume this object to get the underlying vector
    pub fn into_vec(self) -> Vec<ObjCoords> {
        self.path
    }
}

///Enum to help us remember if a point has been added to a path or not
#[derive(Clone)]
pub enum PathStatus {
    OnPath(ObjCoords),
    OffPath(ObjCoords),
}

///Struct for organizing regression results
#[derive(Debug)]
pub struct RegResult {
    v1: f64,
    v2: f64,
    //rms_error: f64,
    max_error: f64,
    avg_error: f64,
    first_point: ObjCoords,
}

impl RegResult {
    ///Perform a linear fit of a series of points starting at 'first_point' and going through
    ///`fit_coords`
    pub fn fit_coords(first_point: &ObjCoords, fit_coords: &[ObjCoords]) -> RegResult {
        let coords: Vec<_> = fit_coords
            .iter()
            .map(|o| {
                //we want to return a (t,x,y) tuple for the regression but normalized
                //to first_point (i.e. with t0, x0 and y0 subtracted off
                //have to convert to float before the subtraction so we don't end up with
                //problems due to subtracting usize past 0
                (
                    TryInto::<f64>::try_into(o.t).unwrap()
                        - TryInto::<f64>::try_into(first_point.t).unwrap(),
                    TryInto::<f64>::try_into(o.x).unwrap()
                        - TryInto::<f64>::try_into(first_point.x).unwrap(),
                    TryInto::<f64>::try_into(o.y).unwrap()
                        - TryInto::<f64>::try_into(first_point.y).unwrap(),
                )
            })
            .collect();
        //build matrices for our least squares fit
        //this will be a column vector of the timestamps
        let nrows = coords.len();
        let t: OMatrix<f64, Dyn, Dyn> = OMatrix::from_iterator_generic(
            Dim::from_usize(nrows),
            Dim::from_usize(1),
            coords.clone().into_iter().map(|r| r.0),
        );
        let mut x: Vec<_> = coords.clone().into_iter().map(|r| r.1).collect();
        let mut y: Vec<_> = coords.clone().into_iter().map(|r| r.2).collect();
        //we want a matrix with x as the first column and y as the second
        x.append(&mut y);
        let xy_mat: OMatrix<f64, Dyn, Dyn> = OMatrix::from_iterator_generic(
            Dim::from_usize(nrows),
            Dim::from_usize(2),
            x.into_iter(),
        );
        //now we can get our fit parameters v1 and v2 as a row vector by multiplying
        //pinv(t) by xy_mat
        let v_mat = pinv(t.clone()) * xy_mat.clone();
        //check that this is the correct size
        assert!(v_mat.nrows() == 1, "v_mat should be 1x2");
        assert!(v_mat.ncols() == 2, "v_mat should be 1x2");
        //grab our components
        let v1 = v_mat[(0, 0)];
        let v2 = v_mat[(0, 1)];
        //we can now calculate the per-point error
        let err_mat: OMatrix<f64, Dyn, Dyn> = xy_mat - t * v_mat;
        //per point l2 norm (distance)
        let l2_err: Vec<_> = err_mat
            .row_iter()
            .map(|r| (r[(0, 0)].powi(2) + r[(0, 1)].powi(2)).sqrt())
            .collect();
        let avg_error = l2_err.iter().sum::<f64>() / (l2_err.len() as f64);
        let max_error = l2_err.into_iter().reduce(|e1, e2| e1.max(e2)).unwrap();
        //let rms_error:f64 = l2_err.clone().into_iter().sum();

        RegResult {
            v1,
            v2,
            max_error,
            avg_error,
            first_point: first_point.clone(),
        }
    }
    ///Get the predicted (x,y) coordinates of this fit at time `t`
    pub fn predict_coords(&self, t: f32) -> (f64, f64) {
        let t_prime = Into::<f64>::into(t - self.first_point.t);
        let pred_x = Into::<f64>::into(self.first_point.x) + self.v1 * t_prime;
        let pred_y = Into::<f64>::into(self.first_point.y) + self.v2 * t_prime;
        (pred_x, pred_y)
    }
    ///Get the L2 error (distance) between `point` and this [RegResult]
    pub fn get_error(&self, point: &ObjCoords) -> f64 {
        //calculate where our fit would place a point at p.t
        let (pred_x, pred_y) = self.predict_coords(point.t);
        //calculate the error in terms of distance
        ((pred_x - Into::<f64>::into(point.x)).powi(2)
            + (pred_y - Into::<f64>::into(point.y)).powi(2))
        .sqrt()
    }
}

///Divide a set of [ObjCoords] into paths and append them to `paths`
///This function will modify the input object list to mark the objects successfully added to paths
///# Arguments
///- `min_points` is the minimum number of coordinates along a path (i.e. how many times we expect to catch an object).
///- `time_window` is the amount forward in time to check for initial path construction.
///- `tolerance` is the error allowed (per point) to be considered on our (linear) path
pub fn track_paths(
    objs: &mut Vec<PathStatus>,
    paths: &mut Vec<ObjPath>,
    min_points: usize,
    time_window: f32,
    tolerance: f32,
) {
    let mut cur_index = 0;
    while cur_index < objs.len() {
        if let PathStatus::OffPath(first_point) = objs[cur_index].clone() {
            //if first_point is OnPath we don't need to worry about it
            //find the first index of our time_window (i.e., the first object to
            //have a timestamp greater than the object at cur_index
            let mut win_min: usize = cur_index;
            for i in (cur_index + 1)..objs.len() {
                let o = match &objs[i] {
                    PathStatus::OnPath(o) => o,
                    PathStatus::OffPath(o) => o,
                };
                if o.t == first_point.t {
                    //this point can't be our object as it's visible in the same frame
                    //cur_index += 1;
                    continue;
                } else {
                    win_min = i;
                    break;
                }
            }
            //find the last index inside our time_window
            let mut win_max: usize = cur_index;
            for i in win_min..objs.len() {
                let o = match &objs[i] {
                    PathStatus::OnPath(o) => o,
                    PathStatus::OffPath(o) => o,
                };
                if (o.t - first_point.t) <= time_window {
                    //this point is inside the window
                    win_max = i;
                } else {
                    break;
                }
            }
            //now, for all unique combinations of objects in our window, which `min_points` set of objects is in the straightest line?
            //println!("win_min:{}, win_max:{}, cur_index:{}",win_min,win_max,cur_index);
            let mut i_combos: Vec<_> = (win_min..win_max)
                .filter(|i| {
                    //only interested in points not already on a path
                    match objs[*i] {
                        PathStatus::OffPath(_) => true,
                        PathStatus::OnPath(_) => false,
                    }
                })
                //our point of interest counts as one
                .combinations(min_points - 1)
                .collect();
            //println!("i_combos before filter {:?}", i_combos);
            //To be the same cell, all of the timestamps on each of the frames must be different.
            i_combos = i_combos
                .into_iter()
                .filter(|c| {
                    //get the timestamps of each frame
                    let mut t_vec: Vec<_> = c
                        .iter()
                        .map(|i| {
                            //need to unwrap our ObjCoords object
                            let PathStatus::OffPath(o) = &objs[*i] else {
                                panic!("we should have removed all OnPath entries");
                            };
                            o.t
                        })
                        .collect();
                    //I think t_vec should be sorted, but we're going to sort it anyways
                    //we need this partial_cmp shenanigans because floats can be NaN and they therefore implement PartialOrd rather than Ord
                    t_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    //our goal here is to check for duplicates. if the length changes after dedup, some of these frames are duplicates and we don't want this combo
                    let old_len = t_vec.len();
                    t_vec.dedup();
                    //if this is false, filter will drop this combination
                    old_len == t_vec.len()
                })
                .collect();
            //fit a linear model to each set of points in i_combos, and choose the one with the lowest error
            //println!("i_combos after filter {:?}", i_combos);
            //we will map i_combos into a vec of RegResult
            let mut fits: Vec<_> = i_combos
                .into_iter()
                .map(|c| {
                    //println!("this combo {:?}",c);
                    //convert our list of indices into a Vec of ObjCoords
                    let coords: Vec<_> = c
                        .iter()
                        .map(|i| {
                            //i is a reference to the index in objs corresponding to this object.
                            //we want to return a (t,x,y) tuple for the regression but normalized
                            //to first_point (i.e. with t0, x0 and y0 subtracted off
                            let PathStatus::OffPath(o) = &objs[*i] else {
                                panic!("we should have removed all OnPath entries");
                            };
                            //println!("time diff: {}",o.t - first_point.t);
                            return o.clone();
                        })
                        .collect();
                    RegResult::fit_coords(&first_point, &coords)
                })
                .collect();
            fits.sort_by(|p1, p2| {
                if p1.max_error < p2.max_error {
                    Ordering::Less
                } else if p1.max_error > p2.max_error {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            if fits.len() == 0 {
                cur_index += 1;
                continue;
            }
            if fits[0].max_error < tolerance.into() {
                //println!("found fit");
                //println!("fits: {:?}",fits);
                //Take the best fit, attempt to extend it, and add it to our output
                let best_fit = &fits[0];
                //create a new ObjPath to store everything in
                let mut this_path_vec = Vec::<ObjCoords>::new();
                let mut path_err: f64 = 0.0;
                //mark first_point as OnPath and add to this_path
                objs[cur_index] = PathStatus::OnPath(first_point.clone());
                this_path_vec.push(first_point.clone());
                //check all remaining points which are OffPath to see if they lie on this fit
                for inner_index in win_min..objs.len() {
                    if let PathStatus::OffPath(p) = objs[inner_index].clone() {
                        //calculate where our fit would place a point at p.t
                        let err = best_fit.get_error(&p);
                        //if this error is less than our tolerance, add this point to our paths
                        //and mark it as OnPath
                        if err < tolerance.into() {
                            objs[inner_index] = PathStatus::OnPath(p.clone());
                            this_path_vec.push(p);
                            if err > path_err {
                                path_err = err
                            }
                        }
                    }
                }
                paths.push(ObjPath {
                    path: this_path_vec,
                    err: path_err,
                    v1: best_fit.v1,
                    v2: best_fit.v2,
                });
            }
        }
        cur_index += 1;
    }
}

const COLOR_PALETTE: [[u8; 3]; 5] = [
    [100, 143, 255],
    [254, 97, 0],
    [120, 94, 240],
    [255, 176, 0],
    [220, 38, 127],
];

///Build debug output when provided a .recbudd file containing the image data, a directory to store
///the output, and the arguments to [find_objects] and [track_paths]
pub fn debug_images(
    recbudd_path: PathBuf,
    out_dir: PathBuf,
    //find_objects args
    bg: &GrayImage,
    blur: f32,
    threshold_value: u8,
    min_obj_area: u64,
    //track_paths args
    min_points: usize,
    time_window: f32,
    tolerance: f32,
) -> Vec<ObjPath> {
    //create our output directory
    let b = DirBuilder::new();
    b.create(&out_dir)
        .expect("couldn't create output directory");
    //open our recbudd file
    let f = File::open(&recbudd_path).expect("couldn't open file");
    let mut reader = BufReader::new(f);
    let mut oc = Vec::<ObjCoords>::new();
    let mut framenum = 1;
    loop {
        match ciborium::from_reader::<recbudd::RecFrame, &mut BufReader<File>>(&mut reader) {
            Ok(rec_frame) => {
                //let timestamp = rec_frame.get_timestamp();
                let im = rec_frame.to_image().into_luma8();
                let (_, cent_vec) =
                    find_objects(bg, &im, blur, threshold_value, min_obj_area, false);
                let mut o: Vec<ObjCoords> = cent_vec
                    .into_iter()
                    .map(|c| ObjCoords {
                        x: c.0,
                        y: c.1,
                        t: framenum as f32,
                    })
                    .collect();
                oc.append(&mut o);
                framenum += 1;
            }
            Err(_) => {
                break;
            }
        }
    }
    //track our paths
    let mut objs_pathstatus: Vec<PathStatus> =
        oc.into_iter().map(|o| PathStatus::OffPath(o)).collect();
    let mut paths = Vec::<ObjPath>::new();
    track_paths(
        &mut objs_pathstatus,
        &mut paths,
        min_points,
        time_window,
        tolerance,
    );
    //rewind our recbudd buffer so we can run through again
    reader
        .seek(SeekFrom::Start(0))
        .expect("couldn't rewind recbudd file");
    //now we will reprocess our frames, this time we will build debug output
    //we want a frame number
    let mut framenum = 1.0;
    loop {
        match ciborium::from_reader::<recbudd::RecFrame, &mut BufReader<File>>(&mut reader) {
            Ok(rec_frame) => {
                let timestamp = rec_frame.get_timestamp();
                //we want a luma8 for processing and an rgba for labeling
                let im_dyn = rec_frame.to_image();
                let im = im_dyn.clone().into_luma8();
                let mut label_im = im_dyn.into_rgba8();
                let (proc, _) = find_objects(bg, &im, blur, threshold_value, min_obj_area, true);
                //now we want to generate a new overlay on label_im which shows all the paths
                //which are 'live' during this time
                let live_path_indices = (0..paths.len()).filter(|i| {
                    let this_path = &paths[*i];
                    //test if this path contains points which occur at or before `framenum`
                    let starts_before = (*this_path).path.iter().any(|c| c.t <= framenum);
                    //test if this path contains points which occur at or after `framenum`
                    let ends_after = (*this_path).path.iter().any(|c| c.t >= framenum);
                    //this path is 'live' if...
                    starts_before && ends_after
                });
                //now, for each live path, we want to draw points and lines connecting all
                //coordinates that have happened up until now
                let im_width = label_im.width();
                let im_height = label_im.height();
                for path_index in live_path_indices {
                    //alternate what color we use
                    let line_color = COLOR_PALETTE[path_index % COLOR_PALETTE.len()];
                    let coords: Vec<_> = paths[path_index]
                        .path
                        .iter()
                        .filter(|coord| coord.t <= framenum)
                        .collect();
                    //println!("path_index: {}, coords.len(): {}",path_index,coords.len());
                    //println!("{:?}",coords);
                    if coords.len() < 1 {
                        //nothing to draw
                        continue;
                    }
                    //draw the first point
                    overlay(
                        &mut label_im,
                        &circle_overlay(
                            im_width,
                            im_height,
                            10,
                            [coords[0].x, coords[0].y],
                            line_color,
                        ),
                        0,
                        0,
                    );
                    if coords.len() < 2 {
                        //done
                        continue;
                    }
                    //for any additional points, draw them and connect with a line
                    for j in 1..coords.len() {
                        let line_start = [coords[j - 1].x, coords[j - 1].y];
                        let line_end = [coords[j].x, coords[j].y];
                        overlay(
                            &mut label_im,
                            &line_overlay(im_width, im_height, line_start, line_end, 5, line_color),
                            0,
                            0,
                        );
                        overlay(
                            &mut label_im,
                            &circle_overlay(im_width, im_height, 10, line_end, line_color),
                            0,
                            0,
                        );
                    }
                }
                //this frame should now be all labeled. concatenate to proc and save in our folder
                let out_im = hcat_image(&[&proc.unwrap(), &label_im]);
                let mut im_path = out_dir.clone();
                im_path.push(format!("frame{}_time{}.png", framenum, timestamp));
                framenum += 1.0;
                out_im.save(im_path).expect("couldn't save image");
            }
            Err(_) => {
                break;
            }
        }
    }
    return paths;
}
