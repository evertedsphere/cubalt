// TODO autogenerate text like "inserrt pair" 
// perhaps method defn includes this logic
#![cfg(all(
    target_feature = "sse",
    target_feature = "sse2",
    target_feature = "bmi1",
    target_feature = "bmi2",
    target_feature = "sse4.1",
    target_feature = "avx",
    target_feature = "avx2",
    target_arch = "x86_64",
))]
#![allow(dead_code)]
pub mod types;
#[macro_use]
pub mod macros;
pub mod avx2;
pub mod cube;
pub mod sse;

use cube::Cube;

pub fn ppr(cube: &Cube) {
    println!("{:?}", cube.0);
    for edge in cube.edges() {
        print!("{:?} | ", edge.0);
    }
    println!();
    for corner in cube.corners() {
        print!("{:?} | ", corner.0);
    }
    println!();
}

pub fn toplevel() {
    println!("toplevel: hi");

    let cube = Cube::identity();
    ppr(&cube);

    let cube_inv = !cube;
    ppr(&cube_inv);

    let mut c = cube;
    ppr(&c);
    c = c * Cube::M_U();
    ppr(&c);
    c = c * Cube::M_U();
    ppr(&c);
    c = c * Cube::M_U();
    ppr(&c);
    c = c * Cube::M_U();
    ppr(&c);
}
