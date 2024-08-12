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
    fn get_centroid(&self) -> (u64, u64) {
        (self.sum_x / self.num_pix, self.sum_y / self.num_pix)
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

///Detect objects
pub fn find_objects(
    bg: &GrayImage,
    f: &GrayImage,
    blur: f32,
    threshold_value: u8,
    min_obj_area: u64,
) -> RgbImage {
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
    let mut centroids = Vec::<(u64, u64)>::new();
    for (_, v) in objs.iter() {
        if v.num_pix >= min_obj_area.into() {
            centroids.push(v.get_centroid())
        }
    }
    println!("centroid list: {:?}", centroids);
    let debug_im =
        DynamicImage::ImageLuma8(hcat_image(&[&f, &bg, &diff, &blurred, &thresh])).into_rgb8();
    let mut labeled_im = DynamicImage::ImageLuma8(f.clone()).into_rgb8();
    let rb = red_box();
    for c in centroids {
        overlay(
            &mut labeled_im,
            &rb,
            c.0.try_into().unwrap(),
            c.1.try_into().unwrap(),
        );
    }
    hcat_image(&[&debug_im, &labeled_im])
}
