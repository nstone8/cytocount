use cytocount::*;

fn main() {
    //(x,y,t)
    let first_point = ObjCoords{x:30,y:20,t:2 as f32};
    let oc:Vec<_> = [(32,32,3),
	      (33,42,4),
	      (32,50,5),
	      (34,62,6)].iter().map(|c| {
		  ObjCoords{x:c.0,y:c.1,t: c.2 as f32}
	      }).collect();
    let fit = RegResult::fit_coords(&first_point,&oc);
    println!("{:?}",fit);
    let err_vec:Vec<_> = oc.clone().iter().map(|c| {
	fit.get_error(&c)
    }).collect();
    println!("per point error: {:?}",err_vec);
    let pred_vec:Vec<_> = oc.iter().map(|c| {
	fit.predict_coords(c.t)
    }).collect();
    println!("predicted points: {:?}",pred_vec);
}
