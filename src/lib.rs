use imageproc::{image,
		filter::gaussian_blur_f32,
		definitions::Image,
		map::map_pixels,
		contrast::{threshold,ThresholdType}};
use image::{ImageBuffer,Pixel,GrayImage,imageops::overlay,Luma,Primitive};
use std::ops::Deref;

///helper function for making debug images
pub fn hcat_image<P,C>(v:&[&ImageBuffer<P,C>]) -> Image<P>
where P:Pixel, C:Deref<Target = [P::Subpixel]>{
    let new_h = v.iter().map(|i| i.height()).max().unwrap();
    let new_w = v.iter().map(|i| i.width()).sum();
    let mut new_im = ImageBuffer::<P,Vec<P::Subpixel>>::new(new_w,new_h);
    let mut cur_x:u32 = 0;
    for im in v {
	overlay(&mut new_im,*im,cur_x.into(),0);
	cur_x += im.width();
    }
    return new_im
}

///subtract one image from another (useful for removing background).
///`image_diff(a,b)` performs `a-b`
pub fn image_diff<T>(a:&Image<Luma<T>>, b:&Image<Luma<T>>) -> Image<Luma<T>>
where T:Primitive{
    map_pixels(a,|x,y,p| {
	//indexing gets us the primitive (numeric value).
	let raw_value = p[0] - b.get_pixel(x,y)[0];
	//a one length array of T implements Into<Luma<T>>
	[raw_value].into()
    })
}

///Detect dark objects on a light background
fn find_objects(bg:&GrayImage,f:&GrayImage,blur:f32,threshold_value:u8) -> GrayImage{
    //subtract off the background
    let diff = image_diff(f,bg);
    //first do a blur to reduce noise
    let blurred = gaussian_blur_f32(f,blur);
    //now threshold
    let thresh = threshold(&blurred,threshold_value,ThresholdType::Binary);
    //filter small objects

    //extract coordinates of object centroids
    hcat_image(&[f,&diff,&blurred,&thresh])
}
