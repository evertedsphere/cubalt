#![allow(non_snake_case)]
use crate::avx2;
use crate::sse;
use crate::types::*;
use std::arch::x86_64::*;

/// The basic SIMD-friendly cube representation.
///
/// Low 128-bit lane:
///   4 U-face edges
///   4 D-face edges
///   4 E-slice edges
///   4 (unused)
///
/// High 128-bit lane:
///   4 U-face corners
///   4 D-face corners
///   8 (unused)
///
/// Edge values (8 bits):
///   ---OEEEE
///   - = unused (zero)
///   O = orientation
///   E = edge index (0..=11)
///
/// Corner values (8 bits):
///   --OO-CCC
///   - = unused (zero)
///   O = orientation (0..=2)
///   C = corner index (0..=7)
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Cube(pub m256i);

/// The low 128-bit lane of the m256 that stores edge state.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct EdgeLane(m128i);

/// The high 128-bit lane of the m256 that stores corner state.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct CornerLane(m128i);

/// A single edge state.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Edge(pub u8);

/// A single corner state.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Corner(pub u8);

impl Cube {
    #[inline(always)]
    pub fn identity() -> Self {
        Self(avx2::identity())
    }

    pub fn new(corners: u64, edges_high: u64, edges_low: u64) -> Self {
        Self(unsafe {
            avx2::literal(
                std::mem::transmute(corners),
                std::mem::transmute(edges_high),
                std::mem::transmute(edges_low),
            )
        })
    }

    fn from_raw_m256(v: m256i) -> Self {
        Self(v)
    }

    /// Parity of the edge + corner permutation
    #[inline(always)]
    pub fn parity(&self) -> bool {
        avx2::parity(self.0)
    }

    pub fn edge_bitmask(&self, bit: u8) -> u32 {
        unsafe {
            std::mem::transmute::<i32, u32>(avx2::bitmask(self.0, bit as i32))
                & 0xffff
        }
    }

    // this can return a u16
    pub fn corner_bitmask(&self, bit: u8) -> u32 {
        unsafe {
            std::mem::transmute::<i32, u32>(avx2::bitmask(self.0, bit as i32))
                >> 16
        }
    }

    pub fn xor_edge_orient(&mut self, eori: Eori) {
        self.0 = avx2::xor_edge_orient(self.0, eori);
    }

    pub fn corner_orient(&self) -> Cori {
        sse::corner_orient(self.corner_lane_ref().0)
    }

    pub fn corner_orient_raw(&self) -> Cori {
        avx2::corner_orient_raw(self.0)
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self(avx2::compose(self.0, other.0))
    }

    pub fn compose_mirror(&self, other: &Self) -> Self {
        Self(avx2::compose_mirror(self.0, other.0))
    }

    pub fn invert(&self) -> Self {
        Cube::from_raw_m256(avx2::invert(self.0))
    }
}

impl std::ops::Not for Cube {
    type Output = Self;
    fn not(self) -> Self {
        self.invert()
    }
}

impl std::ops::Mul for Cube {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.compose(&other)
    }
}

// Edge/corner accessors
impl Cube {
    /// uint8_t *edge = reinterpret_cast<uint8_t*>(&ev());
    #[inline(always)]
    pub fn edges(&self) -> &[Edge] {
        unsafe {
            let edge_lane = self.edge_lane_ref();
            let edge_arr =
                std::mem::transmute::<&EdgeLane, &[Edge; 16]>(&edge_lane);
            &edge_arr[0..=11]
        }
    }

    /// uint8_t *edge = reinterpret_cast<uint8_t*>(&ev());
    #[inline(always)]
    pub fn edges_mut(&mut self) -> &mut [Edge] {
        unsafe {
            let mut edge_lane = self.edge_lane_ref_mut();
            let edge_arr = std::mem::transmute::<&mut EdgeLane, &mut [Edge; 16]>(
                &mut edge_lane,
            );
            &mut edge_arr[0..=11]
        }
    }

    /// __m128i ev() const
    #[inline(always)]
    pub fn edge_lane_ref(&self) -> &EdgeLane {
        unsafe {
            let arr = std::mem::transmute::<&m256i, &[m128i; 2]>(&self.0);
            let ret = std::mem::transmute::<&m128i, &EdgeLane>(&arr[0]);
            ret
        }
    }

    /// __m128i& ev()
    #[inline(always)]
    pub fn edge_lane_ref_mut(&mut self) -> &mut EdgeLane {
        unsafe {
            let arr =
                std::mem::transmute::<&mut m256i, &mut [m128i; 2]>(&mut self.0);
            let ret =
                std::mem::transmute::<&mut m128i, &mut EdgeLane>(&mut arr[0]);
            ret
        }
    }

    /// uint8_t *corner = reinterpret_cast<uint8_t*>(&cv());
    #[inline(always)]
    pub fn corners(&self) -> &[Corner] {
        unsafe {
            // doesn't work
            // let corner_lane = self.corner_lane().0;
            let corner_lane = self.corner_lane_ref();
            let corner_arr =
                std::mem::transmute::<&CornerLane, &[Corner; 16]>(&corner_lane);
            &corner_arr[0..=7]
        }
    }

    /// uint8_t *corner = reinterpret_cast<uint8_t*>(&cv());
    #[inline(always)]
    pub fn corners_mut(&mut self) -> &mut [Corner] {
        unsafe {
            let mut corner_lane = self.corner_lane_ref_mut();
            let corner_arr = std::mem::transmute::<
                &mut CornerLane,
                &mut [Corner; 16],
            >(&mut corner_lane);
            &mut corner_arr[0..=7]
        }
    }

    /// __m128i cv() const
    #[inline(always)]
    pub fn corner_lane_ref(&self) -> &CornerLane {
        unsafe {
            // let arr = std::mem::transmute::<&m256i, &[m128i; 2]>(&self.0);
            // let ret = std::mem::transmute::<&m128i, &CornerLane>(&arr[1]);
            // ret
            let arr = &self.0 as *const _ as *const [m128i; 2];
            let ret = &(*arr)[1] as *const _ as *const CornerLane;
            &*ret
        }
    }

    /// __m128i cv() const
    #[inline(always)]
    pub fn corner_lane_ref_mut(&mut self) -> &mut CornerLane {
        unsafe {
            // let arr = std::mem::transmute::<&mut m256i, &mut [m128i; 2]>(&mut self.0);
            // let ret = std::mem::transmute::<&mut m128i, &mut CornerLane>(&mut arr[1]);
            // ret
            let arr = &mut self.0 as *mut _ as *mut [m128i; 2];
            let ret = &mut (*arr)[1] as *mut _ as *mut CornerLane;
            &mut *ret
        }
    }

    /// A mutable reference to the low half of the m128 that actually stores
    /// corner state.
    /// u64()[2]
    pub fn corners_64_mut(&mut self) -> &mut u64 {
        // let arr = unsafe { std::mem::transmute::<&mut Cube, &mut [u64; 4]>(self) };
        // &mut arr[2]
        unsafe { &mut (*(self as *mut _ as *mut [u64; 4]))[2] }
    }
}

impl Cube {
    /// Set full edge permutation coordinate 0..479001599, and reset edge orientation.
    /// We set only the low 4 bits of every edge and zero the rest, so
    /// the EO (5th bit) of every edge is zeroed.
    pub fn set_edge_perm(&mut self, eperm: Eperm) {
        const FC: [u32; 11] = [
            39916800, 3628800, 362880, 40320, 5040, 720, 120, 24, 6, 2, 1,
        ];
        let mut table: u64 = 0xba9876543210;

        let edges = self.edges_mut();

        // Special case, first iteration does not need "% 12"
        let shift = eperm.0 / FC[0] * 4;
        unsafe {
            edges[0] = Edge(_bextr_u64(table, shift, 4) as u8);
        }
        table ^= (table ^ (table >> 4)) & ((-1i64 as u64) << shift);

        for i in 1..=10 {
            let shift = eperm.0 / FC[i] % (12 - i as u32) * 4;
            unsafe {
                edges[i] = Edge(_bextr_u64(table, shift, 4) as u8);
            }
            table ^= (table ^ (table >> 4)) & ((-1i64 as u64) << shift);
        }

        edges[11] = Edge(table as u8);
    }
}

// -----------------------------------------------------------------------------------------------
// Generated code: move and symmetry maps
// -----------------------------------------------------------------------------------------------

impl Cube {
    /// S_URF3 - 120-degree clockwise rotation on URF-DBL axis (x y)
    pub fn S_URF3() -> Self {
        Self::new(0x1226172321152410, 0x12161410, 0x0a170b1309150811)
    }

    /// The inverse of S_URF3
    pub fn S_URF3i() -> Self {
        Self::new(0x0203000106070405, 0x0a0b0809, 0x0300010207040506)
    }

    /// S_U4   - 90-degree clockwise rotation on U-D axis (y)
    pub fn S_U4() -> Self {
        Self::new(0x0605040702010003, 0x1a19181b, 0x0605040702010003)
    }

    /// S_LR2  - Reflection left to right
    pub fn S_LR2() -> Self {
        Self::new(0x0607040502030001, 0x0a0b0809, 0x0704050603000102)
    }

    /// S_F2 - 180-degree rotation on F-B axis (z2)
    pub fn S_F2() -> Self {
        Self::new(0x0203000106070405, 0x0a0b0809, 0x0300010207040506)
    }

    /// M_U - 90-degree clockwise twist of the U face
    pub fn M_U() -> Self {
        Self::new(0x0706050402010003, 0x0b0a0908, 0x0706050402010003)
    }

    /// Move table:
    /// U, U2, U', R, R2, R', F, F2, F', D, D2, D', L, L2, L', B, B2, B'
    pub fn moves() -> [Self; 18] {
        [
            Cube::new(0x0706050402010003, 0x0b0a0908, 0x0706050402010003),
            Cube::new(0x0706050401000302, 0x0b0a0908, 0x0706050401000302),
            Cube::new(0x0706050400030201, 0x0b0a0908, 0x0706050400030201),
            Cube::new(0x2306051710020124, 0x000a0904, 0x0706050b03020108),
            Cube::new(0x0006050304020107, 0x080a090b, 0x0706050003020104),
            Cube::new(0x2406051017020123, 0x040a0900, 0x070605080302010b),
            Cube::new(0x0706142003022511, 0x0b0a1511, 0x0706180403021900),
            Cube::new(0x0706000103020405, 0x0b0a0809, 0x0706010403020500),
            Cube::new(0x0706112503022014, 0x0b0a1115, 0x0706190403021800),
            Cube::new(0x0407060503020100, 0x0b0a0908, 0x0407060503020100),
            Cube::new(0x0504070603020100, 0x0b0a0908, 0x0504070603020100),
            Cube::new(0x0605040703020100, 0x0b0a0908, 0x0605040703020100),
            Cube::new(0x0715210403261200, 0x0b060208, 0x07090504030a0100),
            Cube::new(0x0701020403050600, 0x0b090a08, 0x0702050403060100),
            Cube::new(0x0712260403211500, 0x0b020608, 0x070a050403090100),
            Cube::new(0x1622050427130100, 0x17130908, 0x1a0605041b020100),
            Cube::new(0x0203050406070100, 0x0a0b0908, 0x0306050407020100),
            Cube::new(0x1327050422160100, 0x13170908, 0x1b0605041a020100),
        ]
    }

    // TODO check this
    pub fn move_sym_6() -> [[u8; 8]; 18] {
        [
            [0, 6, 3, 2, 8, 5, 0, 0],
            [1, 7, 4, 1, 7, 4, 0, 0],
            [2, 8, 5, 0, 6, 3, 0, 0],
            [3, 0, 6, 5, 2, 8, 0, 0],
            [4, 1, 7, 4, 1, 7, 0, 0],
            [5, 2, 8, 3, 0, 6, 0, 0],
            [6, 3, 0, 8, 5, 2, 0, 0],
            [7, 4, 1, 7, 4, 1, 0, 0],
            [8, 5, 2, 6, 3, 0, 0, 0],
            [9, 15, 12, 11, 17, 14, 0, 0],
            [10, 16, 13, 10, 16, 13, 0, 0],
            [11, 17, 14, 9, 15, 12, 0, 0],
            [12, 9, 15, 14, 11, 17, 0, 0],
            [13, 10, 16, 13, 10, 16, 0, 0],
            [14, 11, 17, 12, 9, 15, 0, 0],
            [15, 12, 9, 17, 14, 11, 0, 0],
            [16, 13, 10, 16, 13, 10, 0, 0],
            [17, 14, 11, 15, 12, 9, 0, 0],
        ]
    }

    /// Symmetries (0..47):
    /// S_LR2  (0, 1)
    /// S_F2   (0, 2)
    /// S_U4   (0, 4, 8, 12)
    /// S_URF3 (0, 16, 32)
    pub fn sym() -> [Self; 48] {
        [
            Cube::new(0x0706050403020100, 0x0b0a0908, 0x0706050403020100),
            Cube::new(0x0607040502030001, 0x0a0b0809, 0x0704050603000102),
            Cube::new(0x0203000106070405, 0x0a0b0809, 0x0300010207040506),
            Cube::new(0x0302010007060504, 0x0b0a0908, 0x0302010007060504),
            Cube::new(0x0605040702010003, 0x1a19181b, 0x0605040702010003),
            Cube::new(0x0506070401020300, 0x191a1b18, 0x0607040502030001),
            Cube::new(0x0102030005060704, 0x191a1b18, 0x0203000106070405),
            Cube::new(0x0201000306050407, 0x1a19181b, 0x0201000306050407),
            Cube::new(0x0504070601000302, 0x09080b0a, 0x0504070601000302),
            Cube::new(0x0405060700010203, 0x08090a0b, 0x0506070401020300),
            Cube::new(0x0001020304050607, 0x08090a0b, 0x0102030005060704),
            Cube::new(0x0100030205040706, 0x09080b0a, 0x0100030205040706),
            Cube::new(0x0407060500030201, 0x181b1a19, 0x0407060500030201),
            Cube::new(0x0704050603000102, 0x1b18191a, 0x0405060700010203),
            Cube::new(0x0300010207040506, 0x1b18191a, 0x0001020304050607),
            Cube::new(0x0003020104070605, 0x181b1a19, 0x0003020104070605),
            Cube::new(0x1226172321152410, 0x12161410, 0x0a170b1309150811),
            Cube::new(0x2612231715211024, 0x16121014, 0x0a130b1709110815),
            Cube::new(0x1521102426122317, 0x16121014, 0x091108150a130b17),
            Cube::new(0x2115241012261723, 0x12161410, 0x091508110a170b13),
            Cube::new(0x2617231215241021, 0x06040002, 0x170b130a15081109),
            Cube::new(0x1726122324152110, 0x04060200, 0x170a130b15091108),
            Cube::new(0x2415211017261223, 0x04060200, 0x15091108170a130b),
            Cube::new(0x1524102126172312, 0x06040002, 0x15081109170b130a),
            Cube::new(0x1723122624102115, 0x14101216, 0x0b130a1708110915),
            Cube::new(0x2317261210241521, 0x10141612, 0x0b170a1308150911),
            Cube::new(0x1024152123172612, 0x10141612, 0x081509110b170a13),
            Cube::new(0x2410211517231226, 0x14101216, 0x081109150b130a17),
            Cube::new(0x2312261710211524, 0x00020604, 0x130a170b11091508),
            Cube::new(0x1223172621102415, 0x02000406, 0x130b170a11081509),
            Cube::new(0x2110241512231726, 0x02000406, 0x11081509130b170a),
            Cube::new(0x1021152423122617, 0x00020604, 0x11091508130a170b),
            Cube::new(0x2516221114271320, 0x05070301, 0x161a1219141b1018),
            Cube::new(0x1625112227142013, 0x07050103, 0x1619121a1418101b),
            Cube::new(0x2714201316251122, 0x07050103, 0x1418101b1619121a),
            Cube::new(0x1427132025162211, 0x05070301, 0x141b1018161a1219),
            Cube::new(0x1622112527132014, 0x17131115, 0x1a1219161b101814),
            Cube::new(0x2216251113271420, 0x13171511, 0x1a1619121b141810),
            Cube::new(0x1327142022162511, 0x13171511, 0x1b1418101a161912),
            Cube::new(0x2713201416221125, 0x17131115, 0x1b1018141a121916),
            Cube::new(0x2211251613201427, 0x03010507, 0x1219161a1018141b),
            Cube::new(0x1122162520132714, 0x01030705, 0x121a1619101b1418),
            Cube::new(0x2013271411221625, 0x01030705, 0x101b1418121a1619),
            Cube::new(0x1320142722112516, 0x03010507, 0x1018141b1219161a),
            Cube::new(0x1125162220142713, 0x11151713, 0x19161a1218141b10),
            Cube::new(0x2511221614201327, 0x15111317, 0x19121a1618101b14),
            Cube::new(0x1420132725112216, 0x15111317, 0x18101b1419121a16),
            Cube::new(0x2014271311251622, 0x11151713, 0x18141b1019161a12),
        ]
    }

    /// Inverse symmetry map
    pub fn sym_inv() -> [u8; 48] {
        [
            0, 1, 2, 3, 12, 5, 6, 15, 8, 9, 10, 11, 4, 13, 14, 7, 32, 35, 42,
            41, 20, 21, 28, 29, 34, 33, 40, 43, 22, 23, 30, 31, 16, 25, 24, 17,
            38, 37, 36, 39, 26, 19, 18, 27, 44, 47, 46, 45,
        ]
    }
}
