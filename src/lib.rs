use imageproc::{image,
		filter::gaussian_blur_f32,
		definitions::Image,
		map::map_pixels,
		contrast::{threshold,ThresholdType}};
use image::{ImageBuffer,Pixel,GrayImage,imageops::overlay};
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
///`image_diff(a,b)` performs `abs(a-b)`
fn image_diff(a:&GrayImage, b:&GrayImage) -> GrayImage{
    map_pixels(a,|x,y,p| {
	//indexing gets us the primitive (numeric value).
	let raw_value: i32 = (p[0] as i32) - (b.get_pixel(x,y)[0] as i32);
	//println!("{} - {} = {}",p[0],b.get_pixel(x,y)[0],raw_value);
	let abs_value : u8 = raw_value.abs().try_into().unwrap();
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
///Detect objects
pub fn find_objects(bg:&GrayImage,f:&GrayImage,blur:f32,threshold_value:u8) -> GrayImage{
    //invert our inputs
    //let bg = inv_image(bg);
    //let f = inv_image(f);
    //subtract off the background
    let diff = image_diff(&f,&bg);
    //first do a blur to reduce noise
    let blurred = gaussian_blur_f32(&diff,blur);
    //now threshold
    let thresh = threshold(&blurred,threshold_value,ThresholdType::Binary);
    //filter small objects

    //extract coordinates of object centroids
    hcat_image(&[&f,&bg,&diff,&blurred,&thresh])
}
