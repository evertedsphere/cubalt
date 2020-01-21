#![allow(overflowing_literals)]
use crate::types::*;
use std::arch::x86_64::*;

#[inline(always)]
pub fn identity() -> m128i {
    unsafe { _mm_set_epi64x(0x0706050403020100, 0x0f0e0d0c0b0a0908) }
}

#[inline(always)]
pub fn bitmask(v: m128i, b: i32) -> i32 {
    macro_rules! call {
        ($rhs:expr) => {
            unsafe { _mm_movemask_epi8(_mm_slli_epi32(v, $rhs)) }
        };
    }
    constify_imm8!(7 - b, call)
}

#[inline(always)]
pub fn equals(a: m128i, b: m128i) -> bool {
    unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == -1 }
}

#[inline(always)]
pub fn less_than(a: m128i, b: m128i) -> bool {
    unsafe {
        let gt: i32 = _mm_movemask_epi8(_mm_cmpgt_epi8(a, b));
        let lt: i32 = _mm_movemask_epi8(_mm_cmpgt_epi8(b, a));
        gt < lt
    }
}

#[inline(always)]
pub fn compose_edge(a: m128i, b: m128i) -> m128i {
    unsafe {
        let vperm = _mm_shuffle_epi8(a, b);
        let vori = _mm_and_si128(b, _mm_set1_epi8(0xf0));
        _mm_xor_si128(vperm, vori)
    }
}

#[inline(always)]
pub fn xor_edge_orient(v: m128i, eori: Eori) -> m128i {
    unsafe {
        let mut vori: m128i = _mm_shuffle_epi8(
            _mm_set1_epi32(std::mem::transmute(eori.0)),
            _mm_set_epi64x(0xffffffff01010101, 0),
        );
        vori = _mm_or_si128(vori, _mm_set1_epi64x(!0x8040201008040201));
        vori = _mm_cmpeq_epi8(vori, _mm_set1_epi64x(-1));
        vori = _mm_and_si128(vori, _mm_set1_epi8(0x10));
        _mm_xor_si128(v, vori)
    }
}

#[inline(always)]
pub fn corner_orient(v: m128i) -> Cori {
    unsafe {
        // Mask the corner orientation bits and convert to 16-bit vector
        let mut vorient = _mm_and_si128(v, _mm_set1_epi8(0x30));
        vorient = _mm_unpacklo_epi8(vorient, _mm_setzero_si128());

        // Multiply each corner by its place value, add adjacent pairs
        vorient = _mm_madd_epi16(
            vorient,
            _mm_set_epi16(729, 243, 81, 27, 9, 3, 1, 0),
        );

        // Finish the horizontal sum
        let mut r: i64 =
            _mm_extract_epi64(vorient, 0) + _mm_extract_epi64(vorient, 1);
        r += r >> 32;
        r >>= 4;

        debug_assert!(r < u32::max_value() as i64);

        // FIXME transmute?
        Cori(r as u32)
    }
}
