use image::{imageops::overlay, DynamicImage, GrayImage, ImageBuffer, Pixel, RgbImage};
use imageproc::region_labelling::{connected_components, Connectivity};
use imageproc::{
    contrast::{threshold, ThresholdType},
    definitions::Image,
    filter::gaussian_blur_f32,
    image,
    map::map_pixels,
    rgb_image,
};
use itertools::Itertools;
use moore_penrose::{pinv, Dim, Dyn, OMatrix};
use polars::prelude::{df, DataFrame};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Deref;

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

///Detect objects. This function returns a tuple where the first entry is a `Vec` of object centroids and
///the second entry is an image showing the various phases of the image processing.
pub fn find_objects(
    bg: &GrayImage,
    f: &GrayImage,
    blur: f32,
    threshold_value: u8,
    min_obj_area: u64,
) -> (RgbImage, Vec<(u32, u32)>) {
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
    let debug_im =
        DynamicImage::ImageLuma8(hcat_image(&[&f, &bg, &diff, &blurred, &thresh])).into_rgb8();
    let mut labeled_im = DynamicImage::ImageLuma8(f.clone()).into_rgb8();
    let rb = red_box();
    for c in centroids.iter() {
        overlay(
            &mut labeled_im,
            &rb,
            c.0.try_into().unwrap(),
            c.1.try_into().unwrap(),
        );
    }
    return (hcat_image(&[&debug_im, &labeled_im]), centroids);
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
}

impl ObjPath {
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
struct RegResult {
    v1: f64,
    v2: f64,
    //rms_error: f64,
    max_error: f64,
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
            let mut win_min: usize = cur_index + 1;
            for i in cur_index..objs.len() {
                let o = match &objs[i] {
                    PathStatus::OnPath(o) => o,
                    PathStatus::OffPath(o) => o,
                };
                if o.t == first_point.t {
                    //this point can't be our object as it's visible in the same frame
                    cur_index += 1;
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
                    let coords: Vec<_> = c
                        .iter()
                        .map(|i| {
                            //i is a reference to the index in objs corresponding to this object.
                            //we want to return a (t,x,y) tuple for the regression but normalized
                            //to first_point (i.e. with t0, x0 and y0 subtracted off
                            let PathStatus::OffPath(o) = &objs[*i] else {
                                panic!("we should have removed all OnPath entries");
                            };
                            (
                                TryInto::<f64>::try_into(o.t - first_point.t).unwrap(),
                                TryInto::<f64>::try_into(o.x - first_point.x).unwrap(),
                                TryInto::<f64>::try_into(o.y - first_point.y).unwrap(),
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
                    let max_error = l2_err.into_iter().reduce(|e1, e2| e1.max(e2)).unwrap();
                    //let rms_error:f64 = l2_err.clone().into_iter().sum();

                    RegResult { v1, v2, max_error }
                })
                .collect();
            fits.sort_by(|p1, p2| {
                if p1.max_error < p2.max_error {
                    Ordering::Less
                } else if p1.max_error < p2.max_error {
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
                //Take the best fit, attempt to extend it, and add it to our output
                let best_fit = &fits[0];
                //create a new ObjPath to store everything in
                let mut this_path = ObjPath::new();
                //mark first_point as OnPath and add to this_path
                objs[cur_index] = PathStatus::OnPath(first_point.clone());
                this_path.push(first_point.clone());
                //check all remaining points which are OffPath to see if they lie on this fit
                for inner_index in cur_index + 1..objs.len() {
                    if let PathStatus::OffPath(p) = objs[inner_index].clone() {
                        //calculate where our fit would place a point at p.t
                        let t_prime = Into::<f64>::into(p.t - first_point.t);
                        let pred_x = Into::<f64>::into(first_point.x) + best_fit.v1 * t_prime;
                        let pred_y = Into::<f64>::into(first_point.y) + best_fit.v2 * t_prime;
                        //calculate the error in terms of distance
                        let err = ((pred_x - Into::<f64>::into(p.x)).powi(2)
                            + (pred_y - Into::<f64>::into(p.y)).powi(2))
                        .sqrt();
                        //if this error is less than our tolerance, add this point to our paths
                        //and mark it as OnPath
                        if err < tolerance.into() {
                            objs[inner_index] = PathStatus::OnPath(p.clone());
                            this_path.push(p);
                        }
                    }
                }
                paths.push(this_path);
            }
        }
        cur_index += 1;
    }
}
