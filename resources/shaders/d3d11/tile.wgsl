// pathfinder/resources/shaders/d3d11/tile.wgsl
//
// Copyright © 2026 The Pathfinder Project Developers. [cite: 10]
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. [cite: 11]
//
// This file may not be copied, modified, or distributed
// except according to those terms. [cite: 12]

struct Globals {
    uClearColor: vec4<f32>,
    uLoadAction: i32,
    uPad0: i32,
    uPad1: i32,
    uPad2: i32,
    uTileSize: vec2<f32>,
    uTextureMetadataSize: vec2<f32>,
    uFramebufferSize: vec2<f32>,
    uFramebufferTileSize: vec2<i32>,
    uMaskTextureSize0: vec2<f32>,
    uColorTextureSize0: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var uDestImage: texture_storage_2d<rgba8unorm, read_write>;
@group(2) @binding(0) var uTextureMetadata: texture_2d<f32>;
@group(2) @binding(1) var uZBuffer: texture_2d<f32>;
@group(2) @binding(2) var uColorTexture0: texture_2d<f32>;
@group(2) @binding(3) var uMaskTexture0: texture_2d<f32>;
@group(2) @binding(4) var uGammaLUT: texture_2d<f32>;
@group(2) @binding(5) var smp: sampler;

struct Tile {
    next_tile_id: i32,
    first_fill_id: i32,
    backdrop_alpha: u32,
    ctrl_backdrop: u32,
};
struct Tiles {
    data: array<Tile>,
};
@group(3) @binding(0) var<storage, read> bTiles: Tiles;

struct FirstTileMap {
    data: array<i32>,
};
@group(3) @binding(1) var<storage, read> bFirstTileMap: FirstTileMap;

const LOAD_ACTION_CLEAR: i32 = 0;
const LOAD_ACTION_LOAD: i32 = 1;

const FRAC_6_PI: f32 = 1.9098593171027443;
const FRAC_PI_3: f32 = 1.0471975511965976;

const TILE_CTRL_MASK_MASK: u32 = 0x3u;
const TILE_CTRL_MASK_WINDING: u32 = 0x1u;
const TILE_CTRL_MASK_EVEN_ODD: u32 = 0x2u;

const TILE_CTRL_MASK_0_SHIFT: u32 = 0u;

const COMBINER_CTRL_COLOR_COMBINE_MASK: i32 = 0x3;
const COMBINER_CTRL_COLOR_COMBINE_SRC_IN: i32 = 0x1;
const COMBINER_CTRL_COLOR_COMBINE_DEST_IN: i32 = 0x2;

const COMBINER_CTRL_FILTER_MASK: i32 = 0xf;
const COMBINER_CTRL_FILTER_RADIAL_GRADIENT: i32 = 0x1;
const COMBINER_CTRL_FILTER_TEXT: i32 = 0x2;
const COMBINER_CTRL_FILTER_BLUR: i32 = 0x3;
const COMBINER_CTRL_FILTER_COLOR_MATRIX: i32 = 0x4;

const COMBINER_CTRL_COMPOSITE_MASK: i32 = 0xf;
const COMBINER_CTRL_COMPOSITE_NORMAL: i32 = 0x0;
const COMBINER_CTRL_COMPOSITE_MULTIPLY: i32 = 0x1;
const COMBINER_CTRL_COMPOSITE_SCREEN: i32 = 0x2;
const COMBINER_CTRL_COMPOSITE_OVERLAY: i32 = 0x3;
const COMBINER_CTRL_COMPOSITE_DARKEN: i32 = 0x4;
const COMBINER_CTRL_COMPOSITE_LIGHTEN: i32 = 0x5;
const COMBINER_CTRL_COMPOSITE_COLOR_DODGE: i32 = 0x6;
const COMBINER_CTRL_COMPOSITE_COLOR_BURN: i32 = 0x7;
const COMBINER_CTRL_COMPOSITE_HARD_LIGHT: i32 = 0x8;
const COMBINER_CTRL_COMPOSITE_SOFT_LIGHT: i32 = 0x9;
const COMBINER_CTRL_COMPOSITE_DIFFERENCE: i32 = 0xa;
const COMBINER_CTRL_COMPOSITE_EXCLUSION: i32 = 0xb;
const COMBINER_CTRL_COMPOSITE_HUE: i32 = 0xc;
const COMBINER_CTRL_COMPOSITE_SATURATION: i32 = 0xd;
const COMBINER_CTRL_COMPOSITE_COLOR: i32 = 0xe;
const COMBINER_CTRL_COMPOSITE_LUMINOSITY: i32 = 0xf;

const COMBINER_CTRL_COLOR_FILTER_SHIFT: u32 = 4u;
const COMBINER_CTRL_COLOR_COMBINE_SHIFT: u32 = 8u;
const COMBINER_CTRL_COMPOSITE_SHIFT: u32 = 10u;

// Color combining
fn combineColor0(destColor: vec4<f32>, srcColor: vec4<f32>, op: i32) -> vec4<f32> {
    switch (op) {
        case 0x1: { // COMBINER_CTRL_COLOR_COMBINE_SRC_IN [cite: 26]
            return vec4<f32>(srcColor.rgb, srcColor.a * destColor.a); [cite: 26]
        }
        case 0x2: { // COMBINER_CTRL_COLOR_COMBINE_DEST_IN [cite: 27]
            return vec4<f32>(destColor.rgb, srcColor.a * destColor.a); [cite: 27]
        }
        default: { break; }
    }
    return destColor; [cite: 27]
}

// Text filter
fn filterTextSample1Tap(offset: f32, colorTexture: texture_2d<f32>, colorTexCoord: vec2<f32>) -> float {
    return textureSampleLevel(colorTexture, smp, colorTexCoord + vec2<f32>(offset, 0.0), 0.0).r; [cite: 28]
}

// Samples 9 taps around the current pixel. [cite: 29]
fn filterTextSample9Tap(
    outAlphaLeft: ptr<function, vec4<f32>>,
    outAlphaCenter: ptr<function, f32>,
    outAlphaRight: ptr<function, vec4<f32>>,
    colorTexture: texture_2d<f32>,
    colorTexCoord: vec2<f32>,
    kernel: vec4<f32>,
    onePixel: f32
) {
    let wide: bool = kernel.x > 0.0; [cite: 29]
    var leftVal0 = 0.0;
    if (wide) { leftVal0 = filterTextSample1Tap(-4.0 * onePixel, colorTexture, colorTexCoord); } [cite: 30]

    *outAlphaLeft = vec4<f32>(
        leftVal0, [cite: 30]
        filterTextSample1Tap(-3.0 * onePixel, colorTexture, colorTexCoord), [cite: 30]
        filterTextSample1Tap(-2.0 * onePixel, colorTexture, colorTexCoord), [cite: 30]
        filterTextSample1Tap(-1.0 * onePixel, colorTexture, colorTexCoord) [cite: 30]
    );
    *outAlphaCenter = filterTextSample1Tap(0.0, colorTexture, colorTexCoord); [cite: 31]

    var rightVal3 = 0.0;
    if (wide) { rightVal3 = filterTextSample1Tap(4.0 * onePixel, colorTexture, colorTexCoord); } [cite: 31]

    *outAlphaRight = vec4<f32>(
        filterTextSample1Tap(1.0 * onePixel, colorTexture, colorTexCoord), [cite: 31]
        filterTextSample1Tap(2.0 * onePixel, colorTexture, colorTexCoord), [cite: 31]
        filterTextSample1Tap(3.0 * onePixel, colorTexture, colorTexCoord), [cite: 31]
        rightVal3 [cite: 31]
    );
}

fn filterTextConvolve7Tap(alpha0: vec4<f32>, alpha1: vec3<f32>, kernel: vec4<f32>) -> f32 {
    return dot(alpha0, kernel) + dot(alpha1, kernel.zyx); [cite: 32]
}

fn filterTextGammaCorrectChannel(bgColor: f32, fgColor: f32, gammaLUT: texture_2d<f32>) -> f32 {
    return textureSampleLevel(gammaLUT, smp, vec2<f32>(fgColor, 1.0 - bgColor), 0.0).r; [cite: 33]
}

// `fgColor` is in linear space. [cite: 34]
fn filterTextGammaCorrect(bgColor: vec3<f32>, fgColor: vec3<f32>, gammaLUT: texture_2d<f32>) -> vec3<f32> {
    return vec3<f32>(
        filterTextGammaCorrectChannel(bgColor.r, fgColor.r, gammaLUT), [cite: 34]
        filterTextGammaCorrectChannel(bgColor.g, fgColor.g, gammaLUT), [cite: 34]
        filterTextGammaCorrectChannel(bgColor.b, fgColor.b, gammaLUT) [cite: 34]
    );
}

//                | [cite: 35]
//  x          y          z          w [cite: 36]
//  --------------+-------------------------------------------------------- [cite: 36]
//  filterParams0 | kernel[0]  kernel[1]  kernel[2]  kernel[3] [cite: 36, 37]
//  filterParams1 | bgColor.r  bgColor.g  bgColor.b  - [cite: 37]
//  filterParams2 | fgColor.r  fgColor.g  fgColor.b  gammaCorrectionEnabled [cite: 37, 38]
fn filterText(
    colorTexCoord: vec2<f32>,
    colorTexture: texture_2d<f32>,
    gammaLUT: texture_2d<f32>,
    colorTextureSize: vec2<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>,
    filterParams2: vec4<f32>
) -> vec4<f32> {
    // Unpack. [cite: 38]
    let kernel: vec4<f32> = filterParams0; [cite: 39]
    let bgColor: vec3<f32> = filterParams1.rgb; [cite: 39]
    let fgColor: vec3<f32> = filterParams2.rgb; [cite: 39]
    let gammaCorrectionEnabled: bool = filterParams2.a != 0.0; [cite: 39]

    // Apply defringing if necessary. [cite: 40]
    var alpha: vec3<f32>;
    if (kernel.w == 0.0) {
        alpha = textureSampleLevel(colorTexture, smp, colorTexCoord, 0.0).rrr; [cite: 40]
    } else {
        var alphaLeft: vec4<f32>; [cite: 41]
        var alphaRight: vec4<f32>; [cite: 41]
        var alphaCenter: f32; [cite: 41]
        filterTextSample9Tap(&alphaLeft, &alphaCenter, &alphaRight, colorTexture, colorTexCoord, kernel, 1.0 / colorTextureSize.x); [cite: 42]
        let r: f32 = filterTextConvolve7Tap(alphaLeft, vec3<f32>(alphaCenter, alphaRight.xy), kernel); [cite: 43]
        let g: f32 = filterTextConvolve7Tap(vec4<f32>(alphaLeft.yzw, alphaCenter), alphaRight.xyz, kernel); [cite: 43]
        let b: f32 = filterTextConvolve7Tap(vec4<f32>(alphaLeft.zw, alphaCenter, alphaRight.x), alphaRight.yzw, kernel); [cite: 44]
        alpha = vec3<f32>(r, g, b); [cite: 45]
    }

    // Apply gamma correction if necessary. [cite: 45]
    if (gammaCorrectionEnabled) {
        alpha = filterTextGammaCorrect(bgColor, alpha, gammaLUT); [cite: 46]
    }

    // Finish. [cite: 46]
    return vec4<f32>(mix(bgColor, fgColor, alpha), 1.0); [cite: 46]
}

// Other filters [cite: 47]

// This is based on Pixman (MIT license). Copy and pasting the excellent comment [cite: 47, 48]
// from there: [cite: 48]
// [cite: 48]
// Implementation of radial gradients following the PDF specification. [cite: 48, 49]
// See section 8.7.4.5.4 Type 3 (Radial) Shadings of the PDF Reference [cite: 49]
// Manual (PDF 32000-1:2008 at the time of this writing). [cite: 49, 50]
// [cite: 50]
// In the radial gradient problem we are given two circles (c₁,r₁) and [cite: 50]
// (c₂,r₂) that define the gradient itself. [cite: 50, 51]
// [cite: 51]
// Mathematically the gradient can be defined as the family of circles [cite: 51]
// [cite: 51]
//     ((1-t)·c₁ + t·(c₂), (1-t)·r₁ + t·r₂) [cite: 51]
// [cite: 51]
// excluding those circles whose radius would be < 0. When a point [cite: 51]
// belongs to more than one circle, the one with a bigger t is the only [cite: 51]
// one that contributes to its color. When a point does not belong [cite: 51, 52]
// to any of the circles, it is transparent black, i.e. RGBA (0, 0, 0, 0). [cite: 52]
// Further limitations on the range of values for t are imposed when [cite: 52, 53]
// the gradient is not repeated, namely t must belong to [0,1]. [cite: 53]
// [cite: 53]
// The graphical result is the same as drawing the valid (radius > 0) [cite: 53, 54]
// circles with increasing t in [-∞, +∞] (or in [0,1] if the gradient [cite: 54]
// is not repeated) using SOURCE operator composition. [cite: 54]
// [cite: 54]
// It looks like a cone pointing towards the viewer if the ending circle [cite: 54, 55]
// is smaller than the starting one, a cone pointing inside the page if [cite: 55]
// the starting circle is the smaller one and like a cylinder if they [cite: 55]
// have the same radius. [cite: 55]
// [cite: 55]
// What we actually do is, given the point whose color we are interested [cite: 55, 56]
// in, compute the t values for that point, solving for t in: [cite: 56]
// [cite: 56]
//     length((1-t)·c₁ + t·(c₂) - p) = (1-t)·r₁ + t·r₂ [cite: 56]
// [cite: 56]
// Let's rewrite it in a simpler way, by defining some auxiliary [cite: 56]
// variables: [cite: 56]
// [cite: 56]
//     cd = c₂ - c₁ [cite: 56]
//     pd = p - c₁ [cite: 56]
//     dr = r₂ - r₁ [cite: 56]
//     length(t·cd - pd) = r₁ + t·dr [cite: 56]
// [cite: 56]
// which actually means [cite: 56]
// [cite: 56]
//     hypot(t·cdx - pdx, t·cdy - pdy) = r₁ + t·dr [cite: 56, 57]
// [cite: 57]
// or [cite: 57]
// [cite: 57]
//     ⎷((t·cdx - pdx)² + (t·cdy - pdy)²) = r₁ + t·dr. [cite: 57]
// [cite: 58]
// If we impose (as stated earlier) that r₁ + t·dr ≥ 0, it becomes: [cite: 58]
// [cite: 58]
//     (t·cdx - pdx)² + (t·cdy - pdy)² = (r₁ + t·dr)² [cite: 58]
// [cite: 58]
// where we can actually expand the squares and solve for t: [cite: 58]
// [cite: 58]
//     t²cdx² - 2t·cdx·pdx + pdx² + t²cdy² - 2t·cdy·pdy + pdy² = [cite: 58]
//       = r₁² + 2·r₁·t·dr + t²·dr² [cite: 58]
// [cite: 58]
//     (cdx² + cdy² - dr²)t² - 2(cdx·pdx + cdy·pdy + r₁·dr)t + [cite: 58]
//         (pdx² + pdy² - r₁²) = 0 [cite: 58]
// [cite: 58]
//     A = cdx² + cdy² - dr² [cite: 58, 59]
//     B = pdx·cdx + pdy·cdy + r₁·dr [cite: 59]
//     C = pdx² + pdy² - r₁² [cite: 59]
//     At² - 2Bt + C = 0 [cite: 59]
// [cite: 59]
// The solutions (unless the equation degenerates because of A = 0) are: [cite: 59]
// [cite: 59]
//     t = (B ± ⎷(B² - A·C)) / A [cite: 59]
// [cite: 59]
// The solution we are going to prefer is the bigger one, unless the [cite: 59]
// radius associated to it is negative (or it falls outside the valid t [cite: 59]
// range). [cite: 59]
// [cite: 60]
// Additional observations (useful for optimizations): [cite: 60]
// A does not depend on p [cite: 60]
// [cite: 60]
// A < 0 ⟺ one of the two circles completely contains the other one [cite: 60]
//   ⟺ for every p, the radii associated with the two t solutions have [cite: 60]
//       opposite sign [cite: 60]
// [cite: 60]
//                | [cite: 60]
//  x           y           z               w [cite: 61]
//  --------------+----------------------------------------------------- [cite: 61]
//  filterParams0 | lineFrom.x  lineFrom.y  lineVector.x    lineVector.y [cite: 61, 62]
//  filterParams1 | radii.x     radii.y     uvOrigin.x      uvOrigin.y [cite: 62, 63]
//  filterParams2 | -           -           -               - [cite: 63, 64]
fn filterRadialGradient(
    colorTexCoord: vec2<f32>,
    colorTexture: texture_2d<f32>,
    colorTextureSize: vec2<f32>,
    fragCoord: vec2<f32>,
    framebufferSize: vec2<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>
) -> vec4<f32> {
    let lineFrom: vec2<f32> = filterParams0.xy; [cite: 64]
    let lineVector: vec2<f32> = filterParams0.zw; [cite: 64]
    let radii: vec2<f32> = filterParams1.xy; [cite: 65]
    let uvOrigin: vec2<f32> = filterParams1.zw; [cite: 65]

    let dP: vec2<f32> = colorTexCoord - lineFrom; [cite: 65]
    let dC: vec2<f32> = lineVector; [cite: 65]
    let dR: f32 = radii.y - radii.x; [cite: 66]

    let a: f32 = dot(dC, dC) - dR * dR; [cite: 66]
    let b: f32 = dot(dP, dC) + radii.x * dR; [cite: 67]
    let c: f32 = dot(dP, dP) - radii.x * radii.x; [cite: 67]
    let discrim: f32 = b * b - a * c; [cite: 68]

    var color: vec4<f32> = vec4<f32>(0.0); [cite: 68]
    if (discrim != 0.0) { [cite: 69]
        var ts: vec2<f32> = (sqrt(discrim) * vec2<f32>(1.0, -1.0) + vec2<f32>(b)) / vec2<f32>(a); [cite: 69]
        if (ts.x > ts.y) { [cite: 70]
            ts = ts.yx; [cite: 70]
        }
        let t: f32 = select(ts.y, ts.x, ts.x >= 0.0); [cite: 71]
        color = textureSampleLevel(colorTexture, smp, uvOrigin + vec2<f32>(t, 0.0), 0.0); [cite: 71]
    }
    return color; [cite: 72]
}

//                | [cite: 73]
//  x             y             z             w [cite: 73]
//  --------------+---------------------------------------------------- [cite: 73]
//  filterParams0 | srcOffset.x   srcOffset.y   support       - [cite: 73, 74]
//  filterParams1 | gaussCoeff.x  gaussCoeff.y  gaussCoeff.z  - [cite: 74, 75]
//  filterParams2 | -             -                 -             - [cite: 75, 76]
fn filterBlur(
    colorTexCoord: vec2<f32>,
    colorTexture: texture_2d<f32>,
    colorTextureSize: vec2<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>
) -> vec4<f32> {
    // Unpack. [cite: 76]
    let srcOffsetScale: vec2<f32> = filterParams0.xy / colorTextureSize; [cite: 77]
    let support: i32 = i32(filterParams0.z); [cite: 77]
    var gaussCoeff: vec3<f32> = filterParams1.xyz; [cite: 77]

    // Set up our incremental calculation. [cite: 77]
    var gaussSum: f32 = gaussCoeff.x; [cite: 78]
    var color: vec4<f32> = textureSampleLevel(colorTexture, smp, colorTexCoord, 0.0) * gaussCoeff.x; [cite: 78]
    gaussCoeff.x = gaussCoeff.x * gaussCoeff.y; [cite: 78]
    gaussCoeff.y = gaussCoeff.y * gaussCoeff.z; [cite: 78]

    // This is a common trick that lets us use the texture filtering hardware to evaluate two [cite: 79]
    // texels at a time. [cite: 79, 80]
    // The basic principle is that, if c0 and c1 are colors of adjacent texels [cite: 80]
    // and k0 and k1 are arbitrary factors, the formula `k0 * c0 + k1 * c1` is equivalent to [cite: 80]
    // `(k0 + k1) * lerp(c0, c1, k1 / (k0 + k1))`. [cite: 80]
    // Linear interpolation, as performed by the [cite: 80, 81]
    // texturing hardware when sampling adjacent pixels in one direction, evaluates [cite: 81]
    // `lerp(c0, c1, t)` where t is the offset from the texel with color `c0`. [cite: 81, 82]
    // To evaluate the [cite: 82]
    // formula `k0 * c0 + k1 * c1`, therefore, we can use the texture hardware to perform linear [cite: 82]
    // interpolation with `t = k1 / (k0 + k1)`. [cite: 82]
    for (var i: i32 = 1; i <= support; i += 2) { [cite: 83]
        var gaussPartialSum: f32 = gaussCoeff.x; [cite: 83]
        gaussCoeff.x = gaussCoeff.x * gaussCoeff.y; [cite: 84]
        gaussCoeff.y = gaussCoeff.y * gaussCoeff.z; [cite: 84]
        gaussPartialSum += gaussCoeff.x; [cite: 84]

        let srcOffset: vec2<f32> = srcOffsetScale * (f32(i) + gaussCoeff.x / gaussPartialSum); [cite: 84]
        color += (textureSampleLevel(colorTexture, smp, colorTexCoord - srcOffset, 0.0) +
                  textureSampleLevel(colorTexture, smp, colorTexCoord + srcOffset, 0.0)) * gaussPartialSum; [cite: 85]
        gaussSum += 2.0 * gaussPartialSum; [cite: 86]
        gaussCoeff.x = gaussCoeff.x * gaussCoeff.y; [cite: 86]
        gaussCoeff.y = gaussCoeff.y * gaussCoeff.z; [cite: 86]
    }

    // Finish. [cite: 86]
    return color / gaussSum; [cite: 86]
}

fn filterColorMatrix(
    colorTexCoord: vec2<f32>,
    colorTexture: texture_2d<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>,
    filterParams2: vec4<f32>,
    filterParams3: vec4<f32>,
    filterParams4: vec4<f32>
) -> vec4<f32> {
    let srcColor: vec4<f32> = textureSampleLevel(colorTexture, smp, colorTexCoord, 0.0); [cite: 87]
    let colorMatrix: mat4x4<f32> = mat4x4<f32>(filterParams0, filterParams1, filterParams2, filterParams3); [cite: 88]
    return colorMatrix * srcColor + filterParams4; [cite: 88]
}

fn filterNone(colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>) -> vec4<f32> {
    return textureSampleLevel(colorTexture, smp, colorTexCoord, 0.0); [cite: 89]
}

fn filterColor(
    colorTexCoord: vec2<f32>,
    colorTexture: texture_2d<f32>,
    gammaLUT: texture_2d<f32>,
    colorTextureSize: vec2<f32>,
    fragCoord: vec2<f32>,
    framebufferSize: vec2<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>,
    filterParams2: vec4<f32>,
    filterParams3: vec4<f32>,
    filterParams4: vec4<f32>,
    colorFilter: i32
) -> vec4<f32> {
    switch (colorFilter) {
        case 0x1: { // COMBINER_CTRL_FILTER_RADIAL_GRADIENT
            return filterRadialGradient(colorTexCoord, colorTexture, colorTextureSize, fragCoord, framebufferSize, filterParams0, filterParams1); [cite: 90]
        }
        case 0x3: { // COMBINER_CTRL_FILTER_BLUR
            return filterBlur(colorTexCoord, colorTexture, colorTextureSize, filterParams0, filterParams1); [cite: 91]
        }
        case 0x2: { // COMBINER_CTRL_FILTER_TEXT
            return filterText(colorTexCoord, colorTexture, gammaLUT, colorTextureSize, filterParams0, filterParams1, filterParams2); [cite: 92]
        }
        case 0x4: { // COMBINER_CTRL_FILTER_COLOR_MATRIX
            return filterColorMatrix(colorTexCoord, colorTexture, filterParams0, filterParams1, filterParams2, filterParams3, filterParams4); [cite: 93]
        }
        default: { break; }
    }
    return filterNone(colorTexCoord, colorTexture); [cite: 94]
}

// Compositing
fn compositeSelect(cond: vec3<bool>, ifTrue: vec3<f32>, ifFalse: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(ifFalse.x, ifTrue.x, cond.x), [cite: 94]
        select(ifFalse.y, ifTrue.y, cond.y), [cite: 94]
        select(ifFalse.z, ifTrue.z, cond.z) [cite: 94]
    );
}

fn compositeDivide(num: f32, denom: f32) -> f32 {
    return select(0.0, num / denom, denom != 0.0); [cite: 95]
}

fn compositeColorDodge(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    let destZero: vec3<bool> = destColor == vec3<f32>(0.0); [cite: 96]
    let srcOne: vec3<bool> = srcColor == vec3<f32>(1.0); [cite: 96]
    return compositeSelect(destZero, vec3<f32>(0.0), [cite: 96]
           compositeSelect(srcOne, vec3<f32>(1.0), destColor / (vec3<f32>(1.0) - srcColor))); [cite: 97]
}

// HSL to RGB alternative [cite: 98]
fn compositeHSLToRGB(hsl: vec3<f32>) -> vec3<f32> {
    let a: f32 = hsl.y * min(hsl.z, 1.0 - hsl.z); [cite: 98]
    let ks: vec3<f32> = (vec3<f32>(0.0, 8.0, 4.0) + vec3<f32>(hsl.x * FRAC_6_PI)) % 12.0; [cite: 99]
    return vec3<f32>(hsl.z) - clamp(min(ks - vec3<f32>(3.0), vec3<f32>(9.0) - ks), vec3<f32>(-1.0), vec3<f32>(1.0)) * a; [cite: 100]
}

// From RGB [cite: 101]
fn compositeRGBToHSL(rgb: vec3<f32>) -> vec3<f32> {
    let v: f32 = max(max(rgb.r, rgb.g), rgb.b); [cite: 101]
    let xMin: f32 = min(min(rgb.r, rgb.g), rgb.b); [cite: 101]
    let c: f32 = v - xMin; [cite: 102]
    let l: f32 = mix(xMin, v, 0.5); [cite: 102]
    var terms: vec3<f32>;
    if (rgb.r == v) {
        terms = vec3<f32>(0.0, rgb.g, rgb.b); [cite: 102, 103]
    } else if (rgb.g == v) {
        terms = vec3<f32>(2.0, rgb.b, rgb.r); [cite: 103]
    } else {
        terms = vec3<f32>(4.0, rgb.r, rgb.g); [cite: 103]
    }
    let h: f32 = FRAC_PI_3 * compositeDivide(terms.x * c + terms.y - terms.z, c); [cite: 104]
    let s: f32 = compositeDivide(c, v); [cite: 104]
    return vec3<f32>(h, s, l); [cite: 105]
}

fn compositeScreen(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    return destColor + srcColor - destColor * srcColor; [cite: 105]
}

fn compositeHardLight(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    return compositeSelect(srcColor <= vec3<f32>(0.5), [cite: 106]
           destColor * vec3<f32>(2.0) * srcColor, [cite: 106]
           compositeScreen(destColor, vec3<f32>(2.0) * srcColor - vec3<f32>(1.0))); [cite: 106]
}

fn compositeSoftLight(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    let darkenedDestColor: vec3<f32> = compositeSelect(destColor <= vec3<f32>(0.25), [cite: 107]
                           ((vec3<f32>(16.0) * destColor - vec3<f32>(12.0)) * destColor + vec3<f32>(4.0)) * destColor, [cite: 107]
                           sqrt(destColor)); [cite: 107]
    let factor: vec3<f32> = compositeSelect(srcColor <= vec3<f32>(0.5), [cite: 108]
                        destColor * (vec3<f32>(1.0) - destColor), [cite: 108]
                        darkenedDestColor - destColor); [cite: 108]
    return destColor + (srcColor * 2.0 - 1.0) * factor; [cite: 109]
}

fn compositeHSL(destColor: vec3<f32>, srcColor: vec3<f32>, op: i32) -> vec3<f32> {
    switch (op) {
        case 0xc: { return vec3<f32>(srcColor.x, destColor.y, destColor.z); } // COMBINER_CTRL_COMPOSITE_HUE [cite: 110]
        case 0xd: { return vec3<f32>(destColor.x, srcColor.y, destColor.z); } // COMBINER_CTRL_COMPOSITE_SATURATION [cite: 111]
        case 0xe: { return vec3<f32>(srcColor.x, srcColor.y, destColor.z); } // COMBINER_CTRL_COMPOSITE_COLOR [cite: 112]
        default:  { return vec3<f32>(destColor.x, destColor.y, srcColor.z); } [cite: 113]
    }
}

fn compositeRGB(destColor: vec3<f32>, srcColor: vec3<f32>, op: i32) -> vec3<f32> {
    switch (op) {
        case 0x1: { return destColor * srcColor; } // COMBINER_CTRL_COMPOSITE_MULTIPLY [cite: 114]
        case 0x2: { return compositeScreen(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_SCREEN [cite: 115]
        case 0x3: { return compositeHardLight(srcColor, destColor); } // COMBINER_CTRL_COMPOSITE_OVERLAY [cite: 116]
        case 0x4: { return min(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_DARKEN [cite: 117]
        case 0x5: { return max(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_LIGHTEN [cite: 118]
        case 0x6: { return compositeColorDodge(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_COLOR_DODGE [cite: 119]
        case 0x7: { return vec3<f32>(1.0) - compositeColorDodge(vec3<f32>(1.0) - destColor, vec3<f32>(1.0) - srcColor); } // COMBINER_CTRL_COMPOSITE_COLOR_BURN [cite: 120]
        case 0x8: { return compositeHardLight(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_HARD_LIGHT [cite: 121]
        case 0x9: { return compositeSoftLight(destColor, srcColor); } // COMBINER_CTRL_COMPOSITE_SOFT_LIGHT [cite: 122]
        case 0xa: { return abs(destColor - srcColor); } // COMBINER_CTRL_COMPOSITE_DIFFERENCE [cite: 123]
        case 0xb: { return destColor + srcColor - vec3<f32>(2.0) * destColor * srcColor; } // COMBINER_CTRL_COMPOSITE_EXCLUSION [cite: 124]
        case 0xc: fallthrough;
        case 0xd: fallthrough;
        case 0xe: fallthrough;
        case 0xf: {
            return compositeHSLToRGB(compositeHSL(compositeRGBToHSL(destColor), compositeRGBToHSL(srcColor), op)); [cite: 125]
        }
        default: { break; }
    }
    return srcColor; [cite: 126]
}

fn composite(srcColor: vec4<f32>, destTexture: texture_2d<f32>, destTextureSize: vec2<f32>, fragCoord: vec2<f32>, op: i32) -> vec4<f32> {
    if (op == COMBINER_CTRL_COMPOSITE_NORMAL) { [cite: 126]
        return srcColor; [cite: 126]
    }
    // FIXME(pcwalton): What should the output alpha be here? [cite: 127]
    let destTexCoord: vec2<f32> = fragCoord / destTextureSize; [cite: 127]
    let destColor: vec4<f32> = textureSampleLevel(destTexture, smp, destTexCoord, 0.0); [cite: 128]
    let blendedRGB: vec3<f32> = compositeRGB(destColor.rgb, srcColor.rgb, op); [cite: 128]
    return vec4<f32>(
        srcColor.a * (1.0 - destColor.a) * srcColor.rgb + srcColor.a * destColor.a * blendedRGB + (1.0 - srcColor.a) * destColor.rgb, [cite: 129]
        1.0 [cite: 129]
    );
}

// Masks
fn sampleMask(maskAlpha: f32, maskTexture: texture_2d<f32>, maskTextureSize: vec2<f32>, maskTexCoord: vec3<f32>, maskCtrl: i32) -> f32 {
    if (maskCtrl == 0) { return maskAlpha; } [cite: 130]
    let maskTexCoordI: vec2<i32> = vec2<i32>(floor(maskTexCoord.xy)); [cite: 131]
    let texel: vec4<f32> = textureSampleLevel(maskTexture, smp, (vec2<f32>(maskTexCoordI / vec2<i32>(1, 4)) + 0.5) / maskTextureSize, 0.0); [cite: 131]
    var coverage: f32 = texel[maskTexCoordI.y % 4] + maskTexCoord.z; [cite: 132]

    if ((maskCtrl & i32(TILE_CTRL_MASK_WINDING)) != 0) { [cite: 132]
        coverage = abs(coverage); [cite: 132]
    } else {
        coverage = 1.0 - abs(1.0 - (coverage % 2.0)); [cite: 133]
    }
    return min(maskAlpha, coverage); [cite: 134]
}

// Main helper function
fn calculateColor(
    fragCoord: vec2<f32>,
    colorTexture0: texture_2d<f32>,
    maskTexture0: texture_2d<f32>,
    destTexture: texture_2d<f32>,
    gammaLUT: texture_2d<f32>,
    colorTextureSize0: vec2<f32>,
    maskTextureSize0: vec2<f32>,
    filterParams0: vec4<f32>,
    filterParams1: vec4<f32>,
    filterParams2: vec4<f32>,
    filterParams3: vec4<f32>,
    filterParams4: vec4<f32>,
    framebufferSize: vec2<f32>,
    ctrl: i32,
    maskTexCoord0: vec3<f32>,
    colorTexCoord0: vec2<f32>,
    baseColor: vec4<f32>,
    tileCtrl: i32
) -> vec4<f32> {
    // Sample mask. [cite: 134]
    let maskCtrl0: i32 = (tileCtrl >> i32(TILE_CTRL_MASK_0_SHIFT)) & i32(TILE_CTRL_MASK_MASK); [cite: 135]
    var maskAlpha: f32 = 1.0; [cite: 135]
    maskAlpha = sampleMask(maskAlpha, maskTexture0, maskTextureSize0, maskTexCoord0, maskCtrl0); [cite: 135]

    // Sample color. [cite: 136]
    var color: vec4<f32> = baseColor; [cite: 136]

    // Get color combine flag. [cite: 136]
    let color0Combine: i32 = (ctrl >> COMBINER_CTRL_COLOR_COMBINE_SHIFT) & COMBINER_CTRL_COLOR_COMBINE_MASK; [cite: 136]
    // Do combining. [cite: 137]
    if (color0Combine != 0) { [cite: 137]
        // Get color filter flag. [cite: 137]
        let color0Filter: i32 = (ctrl >> COMBINER_CTRL_COLOR_FILTER_SHIFT) & COMBINER_CTRL_FILTER_MASK; [cite: 138]

        // Do filtering. [cite: 138]
        let color0: vec4<f32> = filterColor(
            colorTexCoord0, colorTexture0, gammaLUT, colorTextureSize0, fragCoord, framebufferSize, [cite: 139]
            filterParams0, filterParams1, filterParams2, filterParams3, filterParams4, color0Filter [cite: 139]
        );
        color = combineColor0(color, color0, color0Combine); [cite: 140]
    }

    // Apply mask. [cite: 141]
    color.a *= maskAlpha; [cite: 141]

    // Apply composite. [cite: 141]
    let compositeOp: i32 = (ctrl >> COMBINER_CTRL_COMPOSITE_SHIFT) & COMBINER_CTRL_COMPOSITE_MASK; [cite: 141]
    color = composite(color, destTexture, framebufferSize, fragCoord, compositeOp); [cite: 142]

    // Premultiply alpha. [cite: 142]
    color.r *= color.a; [cite: 142]
    color.g *= color.a; [cite: 142]
    color.b *= color.a; [cite: 142]
    return color; [cite: 142]
}

fn fetchUnscaled(srcTexture: texture_2d<f32>, scale: vec2<f32>, originCoord: vec2<f32>, entry: i32) -> vec4<f32> {
    return textureSampleLevel(srcTexture, smp, (originCoord + vec2<f32>(0.5) + vec2<f32>(f32(entry), 0.0)) * scale, 0.0); [cite: 146]
}

fn computeTileVaryings(
    position: vec2<f32>,
    colorEntry: i32,
    textureMetadata: texture_2d<f32>,
    textureMetadataSize: vec2<f32>,
    outColorTexCoord0: ptr<function, vec2<f32>>,
    outBaseColor: ptr<function, vec4<f32>>,
    outFilterParams0: ptr<function, vec4<f32>>,
    outFilterParams1: ptr<function, vec4<f32>>,
    outFilterParams2: ptr<function, vec4<f32>>,
    outFilterParams3: ptr<function, vec4<f32>>,
    outFilterParams4: ptr<function, vec4<f32>>,
    outCtrl: ptr<function, i32>
) {
    let metadataScale: vec2<f32> = vec2<f32>(1.0) / textureMetadataSize; [cite: 147]
    let metadataEntryCoord: vec2<f32> = vec2<f32>(f32(colorEntry % 128 * 10), f32(colorEntry / 128)); [cite: 148]
    let colorTexMatrix0: vec4<f32> = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 0); [cite: 148]
    let colorTexOffsets: vec4<f32> = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 1); [cite: 149]
    let baseColor: vec4<f32>       = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 2); [cite: 149]
    let filterParams0: vec4<f32>   = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 3); [cite: 150]
    let filterParams1: vec4<f32>   = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 4); [cite: 150]
    let filterParams2: vec4<f32>   = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 5); [cite: 151]
    let filterParams3: vec4<f32>   = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 6); [cite: 151]
    let filterParams4: vec4<f32>   = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 7); [cite: 152]
    let extra: vec4<f32>           = fetchUnscaled(textureMetadata, metadataScale, metadataEntryCoord, 8); [cite: 152]

    let matrix: mat2x2<f32> = mat2x2<f32>(colorTexMatrix0.xy, colorTexMatrix0.zw); [cite: 153]
    *outColorTexCoord0 = matrix * position + colorTexOffsets.xy; [cite: 153]
    *outBaseColor = baseColor; [cite: 153]
    *outFilterParams0 = filterParams0; [cite: 153]
    *outFilterParams1 = filterParams1; [cite: 153]
    *outFilterParams2 = filterParams2; [cite: 153]
    *outFilterParams3 = filterParams3; [cite: 154]
    *outFilterParams4 = filterParams4; [cite: 154]
    *outCtrl = i32(extra.x); [cite: 154]
}

fn calculateTileIndex(bufferOffset: u32, tileRect: vec4<u32>, tileCoord: vec2<u32>) -> u32 {
    return bufferOffset + tileCoord.y * (tileRect.z - tileRect.x) + tileCoord.x; [cite: 154]
}

fn toImageCoords(coords: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(coords.x, coords.y); [cite: 155]
}

@compute @workgroup_size(16, 4) [cite: 1]
fn cs_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tileCoord = vec2<i32>(i32(group_id.x), i32(group_id.y)); [cite: 155]
    let firstTileSubCoord = vec2<i32>(i32(local_id.x), i32(local_id.y)) * vec2<i32>(1, 4); [cite: 156]
    let firstFragCoord = tileCoord * vec2<i32>(globals.uTileSize) + firstTileSubCoord; [cite: 156]

    // Quick exit if this is guaranteed to be empty. [cite: 157]
    var tileIndex: i32 = bFirstTileMap.data[u32(tileCoord.x + globals.uFramebufferTileSize.x * tileCoord.y)]; [cite: 157]
    if (tileIndex < 0 && globals.uLoadAction != LOAD_ACTION_CLEAR) { return; } [cite: 158]

    var destColors: array<vec4<f32>, 4>;
    for (var subY: i32 = 0; subY < 4; subY++) { [cite: 159]
        if (globals.uLoadAction == LOAD_ACTION_CLEAR) { [cite: 159]
            destColors[subY] = globals.uClearColor; [cite: 159]
        } else {
            // Not available for GLES. [cite: 160]
            let imageCoords: vec2<i32> = toImageCoords(firstFragCoord + vec2<i32>(0, subY)); [cite: 161]
            destColors[subY] = textureLoad(uDestImage, imageCoords); [cite: 162]
        }
    }

    while (tileIndex >= 0) { [cite: 162]
        let tile = bTiles.data[u32(tileIndex)];

        for (var subY: i32 = 0; subY < 4; subY++) { [cite: 162]
            let tileSubCoord: vec2<i32> = firstTileSubCoord + vec2<i32>(0, subY); [cite: 162]
            let fragCoord: vec2<f32> = vec2<f32>(firstFragCoord + vec2<i32>(0, subY)) + vec2<f32>(0.5); [cite: 163]

            // [ +-8 | +-24 ] note that ALPHA_TILE_ID is signed. [cite: 164]
            let alphaTileIndex: i32 = i32(tile.backdrop_alpha << 8u) >> 8u; [cite: 164]
            let tileControlWord: u32 = tile.ctrl_backdrop; [cite: 164]
            let colorEntry: u32 = tileControlWord & 0xffffu; [cite: 165]
            var tileCtrl: i32 = i32((tileControlWord >> 16u) & 0xffu); [cite: 165]

            var backdrop: i32; [cite: 165]
            var maskTileCoord: vec2<u32>; [cite: 165]

            // alphaTileIndex >= 0 -> alpha tiles. [cite: 166]
            // alphaTileIndex < 0 -> solid tiles. [cite: 166, 167]
            if (alphaTileIndex >= 0) { [cite: 167]
                backdrop = 0; [cite: 167]
                maskTileCoord = vec2<u32>(alphaTileIndex & 0xff, alphaTileIndex >> 8) * vec2<u32>(globals.uTileSize); [cite: 168]

                // Uncomment this to hide alpha tiles. [cite: 168]
                // return; [cite: 168]
            } else {
                // We have no alpha mask. [cite: 169]
                // Clear the mask bits so we don't try to look one up. [cite: 170]
                backdrop = i32(tileControlWord) >> 24; [cite: 170]

                // Handle solid tiles hiden by the even-odd fill rule. [cite: 171]
                if (backdrop != 0) { [cite: 172]
                    let maskCtrl: i32 = (tileCtrl >> i32(TILE_CTRL_MASK_0_SHIFT)) & i32(TILE_CTRL_MASK_MASK); [cite: 172]
                    if ((maskCtrl & i32(TILE_CTRL_MASK_EVEN_ODD)) != 0 && (backdrop & 1) == 0) { [cite: 173]
                        break; // 还原原版 GLSL 这里的 break [cite: 173]
                    }
                }

                maskTileCoord = vec2<u32>(0u); [cite: 174]
                tileCtrl &= ~i32(TILE_CTRL_MASK_MASK << TILE_CTRL_MASK_0_SHIFT); [cite: 175]

                // Uncomment this to hide solid tiles. [cite: 175]
                // return; [cite: 175]
            }

            let maskTexCoord0: vec3<f32> = vec3<f32>(vec2<f32>(maskTileCoord) + vec2<f32>(tileSubCoord), f32(backdrop)); [cite: 176]
            var colorTexCoord0: vec2<f32>; [cite: 177]
            var baseColor: vec4<f32>; [cite: 177]
            var filterParams0: vec4<f32>; [cite: 177]
            var filterParams1: vec4<f32>; [cite: 177]
            var filterParams2: vec4<f32>; [cite: 177]
            var filterParams3: vec4<f32>; [cite: 177]
            var filterParams4: vec4<f32>; [cite: 177]
            var ctrl: i32; [cite: 177]

            computeTileVaryings(
                fragCoord, i32(colorEntry), uTextureMetadata, globals.uTextureMetadataSize,
                &colorTexCoord0, &baseColor, &filterParams0, &filterParams1, &filterParams2, &filterParams3, &filterParams4, &ctrl
            ); [cite: 178, 179]

            // FIXME(pcwalton): The `uColorTexture0` below is a placeholder and needs to be replaced! [cite: 180]
            // 此处一比一还原了原本被略过的外部全规格 calculateColor 函数参数集 [cite: 181]
            let srcColor: vec4<f32> = calculateColor(
                fragCoord, uColorTexture0, uMaskTexture0, uColorTexture0, uGammaLUT,
                globals.uColorTextureSize0, globals.uMaskTextureSize0,
                filterParams0, filterParams1, filterParams2, filterParams3, filterParams4,
                globals.uFramebufferSize, ctrl, maskTexCoord0, colorTexCoord0, baseColor, tileCtrl
            ); [cite: 181, 182, 183]

            destColors[subY] = destColors[subY] * (1.0 - srcColor.a) + srcColor; [cite: 184]
        }

        tileIndex = bTiles.data[u32(tileIndex)].next_tile_id; [cite: 184]
    }

    for (var subY: i32 = 0; subY < 4; subY++) { [cite: 185]
        textureStore(uDestImage, toImageCoords(firstFragCoord + vec2<i32>(0, subY)), destColors[subY]); [cite: 185]
    }
}