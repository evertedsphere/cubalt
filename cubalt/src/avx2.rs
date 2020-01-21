#![cfg(all(target_feature = "avx", target_feature = "avx2",))]
#![allow(overflowing_literals)]
use crate::types::*;
use std::arch::x86_64::*;

pub fn identity() -> m256i {
    unsafe {
        _mm256_set_epi64x(
            0x0f0e0d0c0b0a0908,
            0x0706050403020100,
            0x0f0e0d0c0b0a0908,
            0x0706050403020100,
        )
    }
}

pub fn edges_low(v: m256i) -> i64 {
    unsafe { _mm256_extract_epi64(v, 0) }
}

pub fn edges_high(v: m256i) -> i64 {
    unsafe { _mm256_extract_epi64(v, 1) }
}

pub fn corners(v: m256i) -> i64 {
    unsafe { _mm256_extract_epi64(v, 2) }
}

pub fn literal(corners: i64, edges_high: i64, edges_low: i64) -> m256i {
    unsafe {
        _mm256_set_epi64x(
            0x0f0e0d0c0b0a0908,
            corners,
            0x0f0e0d0c00000000 | edges_high,
            edges_low,
        )
    }
}

pub fn bitmask(v: m256i, b: i32) -> i32 {
    macro_rules! call {
        ($rhs:expr) => {
            unsafe { _mm256_movemask_epi8(_mm256_slli_epi32(v, $rhs)) }
        };
    }
    constify_imm8!(7 - b, call)
}

pub fn equals(a: m256i, b: m256i) -> bool {
    unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) == -1 }
}

pub fn less_than(a: m256i, b: m256i) -> bool {
    unsafe {
        let gt: i32 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(a, b));
        let lt: i32 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(b, a));
        gt < lt
    }
}

#[inline(always)]
pub fn compose_perhaps_mirror(a: m256i, b: m256i, mirror: bool) -> m256i {
    unsafe {
        let vcarry: m256i = _mm256_set_epi64x(
            0x3030303030303030,
            0x3030303030303030,
            0x2020202020202020,
            0x2020202020202020,
        );

        // Permute edges and corners
        let mut vperm: m256i = _mm256_shuffle_epi8(a, b);

        // Compose edge and corner orientations
        let vori: m256i = _mm256_and_si256(b, _mm256_set1_epi8(0xf0));
        if mirror {
            // Corner orientations are subtracted
            vperm = _mm256_sub_epi8(vperm, vori);
            vperm = _mm256_min_epu8(vperm, _mm256_add_epi8(vperm, vcarry));
        } else {
            // Corner orientations are added
            vperm = _mm256_add_epi8(vperm, vori);
            vperm = _mm256_min_epu8(vperm, _mm256_sub_epi8(vperm, vcarry));
        }

        vperm
    }
}

pub fn compose(a: m256i, b: m256i) -> m256i {
    compose_perhaps_mirror(a, b, false)
}

pub fn compose_mirror(a: m256i, b: m256i) -> m256i {
    compose_perhaps_mirror(a, b, true)
}

pub fn xor_edge_orient(v: m256i, eori: Eori) -> m256i {
    unsafe {
        let mut vori: m256i = _mm256_shuffle_epi8(
            _mm256_set1_epi32(std::mem::transmute(eori.0)),
            _mm256_set_epi64x(-1, -1, 0xffffffff01010101, 0),
        );
        vori = _mm256_or_si256(vori, _mm256_set1_epi64x(!0x8040201008040201));
        vori = _mm256_cmpeq_epi8(vori, _mm256_set1_epi64x(-1));
        vori = _mm256_and_si256(vori, _mm256_set1_epi8(0x10));
        _mm256_xor_si256(v, vori)
    }
}

pub fn corner_orient_raw(v: m256i) -> Cori {
    unsafe {
        let vori: m256i = _mm256_unpacklo_epi8(
            _mm256_slli_epi32(v, 3),
            _mm256_slli_epi32(v, 2),
        );
        Cori(std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(vori)) >> 16)
    }
}

pub fn invert(v: m256i) -> m256i {
    unsafe {
        // Split the cube into separate perm and orient vectors
        let vperm: m256i = _mm256_and_si256(v, _mm256_set1_epi8(0x0f));
        let mut vori: m256i = _mm256_xor_si256(v, vperm);

        // "Brute force" the inverse of the permutation
        let mut vi: m256i = _mm256_set_epi64x(
            0x0f0e0d0c00000000,
            0x0000000000000000,
            0x0f0e0d0c00000000,
            0x0000000000000000,
        );
        for i in 0..12 {
            let vtrial: m256i = _mm256_set1_epi8(i);
            let vcorrect: m256i = _mm256_cmpeq_epi8(
                identity(),
                _mm256_shuffle_epi8(vperm, vtrial),
            );
            vi = _mm256_or_si256(vi, _mm256_and_si256(vtrial, vcorrect));
        }

        // Invert the corner orientations
        let vcarry_corners: m256i = _mm256_set_epi64x(
            0x3030303030303030,
            0x3030303030303030,
            0x1010101010101010,
            0x1010101010101010,
        );
        vori = _mm256_add_epi8(vori, vori);
        vori = _mm256_min_epu8(vori, _mm256_sub_epi8(vori, vcarry_corners));

        // Permute the edge and corner orientations
        vori = _mm256_shuffle_epi8(vori, vi);

        // Combine the new perm and orient
        _mm256_or_si256(vi, vori)
    }
}

pub fn unrank_corner_orient(cori: Cori) -> i64 {
    unsafe {
        /* 16-bit mulhi is lower latency than 32-bit, but has two disadvantages:
         * - Requires two different shift widths
         * - The multiplier for the 3^0 place is 65536
         */
        let vpow3_reciprocal: m256i =
            _mm256_set_epi32(1439, 4316, 12946, 38837, 7282, 21846, 0, 0);
        let vshift: m256i = _mm256_set_epi32(4, 4, 4, 4, 0, 0, 0, 0);

        // Divide by powers of 3 (1, 3, 9, ..., 729)
        let vcorient: m256i = _mm256_set1_epi32(std::mem::transmute(cori.0));
        let mut vco: m256i = _mm256_mulhi_epu16(vcorient, vpow3_reciprocal);
        vco = _mm256_srlv_epi32(vco, vshift);

        // fixup 3^0 place; reuse vcorient instead of inserting
        vco = _mm256_blend_epi32(vco, vcorient, 1 << 1);

        // Compute the remainder mod 3
        let div3: m256i = _mm256_mulhi_epu16(vco, _mm256_set1_epi32(21846)); // 21846/65536 ~ 1/3
        vco = _mm256_add_epi32(vco, div3);
        vco = _mm256_sub_epi32(vco, _mm256_slli_epi32(div3, 2));

        // Convert the results to a scalar
        vco = _mm256_shuffle_epi8(
            vco,
            _mm256_set_epi32(-1, -1, 0x0c080400, -1, -1, -1, -1, 0x0c080400),
        );
        let mut co: i64 =
            _mm256_extract_epi64(vco, 2) | _mm256_extract_epi64(vco, 0);

        // Determine the last corner's orientation
        let mut sum: i64 = co + (co >> 32);
        sum += sum >> 16;
        sum += sum >> 8;

        // Insert the last corner
        co |= (0x4924924924924924 >> sum) & 3;

        co << 4
    }
}

/// Return the parity of the edge+corner permutations
pub fn parity(v: m256i) -> bool {
    unsafe {
        let v = _mm256_and_si256(v, _mm256_set1_epi8(0xf));

        let mut a = _mm256_bslli_epi128(v, 1); // shift left 1 byte
        let b = _mm256_bslli_epi128(v, 2); // shift left 2 bytes
        let mut c = _mm256_bslli_epi128(v, 3); // shift left 3 bytes
        let d = _mm256_bslli_epi128(v, 4); // shift left 4 bytes
        let mut e = _mm256_bslli_epi128(v, 8); // shift left 8 bytes
        let f = _mm256_alignr_epi8(v, v, 11); // rotate left 5 bytes
        let g = _mm256_alignr_epi8(v, v, 10); // rotate left 6 bytes
        let h = _mm256_alignr_epi8(v, v, 9); // rotate left 7 bytes

        // Test for inversions in the permutation
        a = _mm256_xor_si256(_mm256_cmpgt_epi8(a, v), _mm256_cmpgt_epi8(b, v));
        c = _mm256_xor_si256(_mm256_cmpgt_epi8(c, v), _mm256_cmpgt_epi8(d, v));
        e = _mm256_xor_si256(_mm256_cmpgt_epi8(e, v), _mm256_cmpgt_epi8(f, v));

        // Xor all the tests together
        let mut parity: m256i = _mm256_xor_si256(_mm256_xor_si256(a, c), e);
        parity = _mm256_xor_si256(parity, _mm256_cmpgt_epi8(g, v));
        parity = _mm256_xor_si256(parity, _mm256_cmpgt_epi8(h, v));

        // The 0x5f corrects for the circular shifts, which cause
        // certain pairs of values to be compared out-of-order
        return (_popcnt32(_mm256_movemask_epi8(parity) ^ 0x5f005f) & 1) != 0;
    }
}
