// pathfinder/resources/shaders/d3d9/tile.wgsl
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Globals {
    uTileSize: vec2<f32>, // Fixed as (16, 16).
    uTextureMetadataSize: vec2<i32>, // Fixed as (1280, 512).
    uZBufferSize: vec2<i32>, // Not used here in fragment shader.
    uMaskTextureSize0: vec2<f32>, // Dynamic as (4096, 1024 * page_count).
    uColorTextureSize0: vec2<f32>,
    uFramebufferSize: vec2<f32>, // Dst framebuffer.
    uTransform: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var uTextureMetadata: texture_2d<f32>;
@group(1) @binding(1) var uZBuffer: texture_2d<f32>;
@group(1) @binding(2) var uColorTexture0: texture_2d<f32>; // Pattern image.
@group(1) @binding(3) var uMaskTexture0: texture_2d<f32>;
@group(1) @binding(4) var uDestTexture: texture_2d<f32>;
@group(1) @binding(5) var uGammaLUT: texture_2d<f32>; // For text.
@group(1) @binding(6) var smp: sampler;

struct VertexInput {
    @location(0) aTileOffset: vec2<u32>, // Tile local coordinates
    @location(1) aTileOrigin: vec2<i32>, // Tile index
    @location(2) aMaskTexCoord0: vec4<u32>,
    @location(3) aPathIndex: i32,
    @location(4) aCtrlBackdrop: vec2<i32>,
    @location(5) aMetadataIndex: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vMaskTexCoord0: vec3<f32>,
    @location(1) vColorTexCoord0: vec2<f32>,
    @location(2) vBaseColor: vec4<f32>,
    @location(3) vTileCtrl: f32,
    @location(4) vFilterParams0: vec4<f32>,
    @location(5) vFilterParams1: vec4<f32>,
    @location(6) vFilterParams2: vec4<f32>,
    @location(7) vFilterParams3: vec4<f32>,
    @location(8) vFilterParams4: vec4<f32>,
    @location(9) vCtrl: f32,
};

const FRAC_6_PI: f32 = 1.9098593171027443;
const FRAC_PI_3: f32 = 1.0471975511965976;

const TILE_CTRL_MASK_MASK: i32 = 0x3;
const TILE_CTRL_MASK_WINDING: i32 = 0x1;
const TILE_CTRL_MASK_EVEN_ODD: i32 = 0x2;

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

/// Fetch data from the metadata texture.
fn fetchUnscaled(srcTexture: texture_2d<f32>, originCoord: vec2<f32>, entry: i32) -> vec4<f32> {
    let pixelCoord = vec2<i32>(i32(originCoord.x) + entry, i32(originCoord.y));
    return textureLoad(srcTexture, pixelCoord, 0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Global tile coordinates.
    let tileOrigin = vec2<f32>(input.aTileOrigin);

    // Local vertex offset, i.e. (0,0), (0,1), (1,1), (1,0).
    let tileOffset = vec2<f32>(input.aTileOffset);

    // Global vertex position.
    let position = (tileOrigin + tileOffset) * globals.uTileSize;

    // Tile culling.
    // --------------------------------------------------
    // Get the UV coordinates of the tile Z value.
    let zValue = textureLoad(uZBuffer, input.aTileOrigin, 0);

    // Note that Z value is packed into a RBGA8 pixel.
    // Unpack it. Compare it with the current path index to see
    // if the current tile is under another opaque tile.
    let unpackedZ = i32(u32(zValue.r * 255.0) | (u32(zValue.g * 255.0) << 8u) | (u32(zValue.b * 255.0) << 16u) | (u32(zValue.a * 255.0) << 24u));
    if (input.aPathIndex < unpackedZ) {
        // Tile culled.
        out.position = vec4<f32>(0.0);
        return out;
    }
    // --------------------------------------------------

    // Global position of the corresponding mask tile.
    let maskTileCoord = vec2<u32>(input.aMaskTexCoord0.x, input.aMaskTexCoord0.y + 256u * input.aMaskTexCoord0.z);
    let maskTexCoord0 = (vec2<f32>(maskTileCoord) + tileOffset) * globals.uTileSize;

    // aMaskTexCoord0.w != 0u means alpha_tile_id is too large (invalid in that case).
    if (input.aCtrlBackdrop.y == 0 && input.aMaskTexCoord0.w != 0u) {
        out.position = vec4<f32>(0.0);
        return out;
    }

    // Pixel coordinates.
    let metadataEntryCoord = vec2<f32>(f32(input.aMetadataIndex % 128u * 10u), f32(input.aMetadataIndex / 128u));

    // Fetch data via texture().
    let colorTexMatrix0 = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 0);
    let colorTexOffsets = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 1);
    let baseColor       = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 2); // Solid color.
    let filterParams0   = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 3);
    let filterParams1   = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 4);
    let filterParams2   = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 5);
    let filterParams3   = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 6);
    let filterParams4   = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 7);
    let extra           = fetchUnscaled(uTextureMetadata, metadataEntryCoord, 8);

    // Set color texture coordinates.
    out.vColorTexCoord0 = mat2x2<f32>(colorTexMatrix0.xy, colorTexMatrix0.zw) * position + colorTexOffsets.xy;

    // Set base color.
    out.vBaseColor = baseColor;

    // Debug
//    out.vBaseColor = vec4<f32>(1.0, 0.0, 0.0, 1.0);

    // Set filter parameters.
    out.vFilterParams0 = filterParams0;
    out.vFilterParams1 = filterParams1;
    out.vFilterParams2 = filterParams2;
    out.vFilterParams3 = filterParams3;
    out.vFilterParams4 = filterParams4;

    // Set blend and composite options.
    let ctrl = i32(extra.x);

    out.vTileCtrl = f32(input.aCtrlBackdrop.x);
    out.vCtrl = f32(ctrl);
    out.vMaskTexCoord0 = vec3<f32>(maskTexCoord0, f32(input.aCtrlBackdrop.y));

    // uTransform converts UV coodinates to screen coodinates.
    let pos = globals.uTransform * vec4<f32>(position, 0.0, 1.0);
    out.position = vec4<f32>(pos.x, pos.y, pos.z, pos.w); // WebGPU Y-axis flip
    return out;
}

// === Fragment Shader Helper Functions ===

// Color combining
fn combineColor0(destColor: vec4<f32>, srcColor: vec4<f32>, op: i32) -> vec4<f32> {
    switch (op) {
        case COMBINER_CTRL_COLOR_COMBINE_SRC_IN: {
            return vec4<f32>(srcColor.rgb, srcColor.a * destColor.a);
        }
        case COMBINER_CTRL_COLOR_COMBINE_DEST_IN: {
            return vec4<f32>(destColor.rgb, srcColor.a * destColor.a);
        }
        default: {}
    }
    return destColor;
}

// Text filter
fn filterTextSample1Tap(offset: f32, colorTexture: texture_2d<f32>, colorTexCoord: vec2<f32>) -> f32 {
    return textureSample(colorTexture, smp, colorTexCoord + vec2<f32>(offset, 0.0)).r;
}

fn filterTextGammaCorrectChannel(bgColor: f32, fgColor: f32, gammaLUT: texture_2d<f32>) -> f32 {
    return textureSample(gammaLUT, smp, vec2<f32>(fgColor, 1.0 - bgColor)).r;
}

fn filterTextGammaCorrect(bgColor: vec3<f32>, fgColor: vec3<f32>, gammaLUT: texture_2d<f32>) -> vec3<f32> {
    return vec3<f32>(
        filterTextGammaCorrectChannel(bgColor.r, fgColor.r, gammaLUT),
        filterTextGammaCorrectChannel(bgColor.g, fgColor.g, gammaLUT),
        filterTextGammaCorrectChannel(bgColor.b, fgColor.b, gammaLUT)
    );
}

fn filterText(
    colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>, gammaLUT: texture_2d<f32>,
    colorTextureSize: vec2<f32>, filterParams0: vec4<f32>, filterParams1: vec4<f32>, filterParams2: vec4<f32>
) -> vec4<f32> {
    let kernel = filterParams0;
    let bgColor = filterParams1.rgb;
    let fgColor = filterParams2.rgb;
    let gammaCorrectionEnabled = filterParams2.a != 0.0;

    var alpha: vec3<f32>;
    if (kernel.w == 0.0) {
        alpha = textureSample(colorTexture, smp, colorTexCoord).rrr;
    } else {
        let onePixel = 1.0 / colorTextureSize.x;
        let wide = kernel.x > 0.0;

        let alphaLeft = vec4<f32>(
            select(0.0, filterTextSample1Tap(-4.0 * onePixel, colorTexture, colorTexCoord), wide),
            filterTextSample1Tap(-3.0 * onePixel, colorTexture, colorTexCoord),
            filterTextSample1Tap(-2.0 * onePixel, colorTexture, colorTexCoord),
            filterTextSample1Tap(-1.0 * onePixel, colorTexture, colorTexCoord)
        );
        let alphaCenter = filterTextSample1Tap(0.0, colorTexture, colorTexCoord);
        let alphaRight = vec4<f32>(
            filterTextSample1Tap(1.0 * onePixel, colorTexture, colorTexCoord),
            filterTextSample1Tap(2.0 * onePixel, colorTexture, colorTexCoord),
            filterTextSample1Tap(3.0 * onePixel, colorTexture, colorTexCoord),
            select(0.0, filterTextSample1Tap(4.0 * onePixel, colorTexture, colorTexCoord), wide)
        );

        let r = dot(alphaLeft, kernel) + dot(vec3<f32>(alphaCenter, alphaRight.xy), kernel.zyx);
        let g = dot(vec4<f32>(alphaLeft.yzw, alphaCenter), kernel) + dot(alphaRight.xyz, kernel.zyx);
        let b = dot(vec4<f32>(alphaLeft.zw, alphaCenter, alphaRight.x), kernel) + dot(alphaRight.yzw, kernel.zyx);
        alpha = vec3<f32>(r, g, b);
    }

    if (gammaCorrectionEnabled) {
        alpha = filterTextGammaCorrect(bgColor, alpha, gammaLUT);
    }

    return vec4<f32>(mix(bgColor, fgColor, alpha), 1.0);
}

fn filterRadialGradient(
    colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>, filterParams0: vec4<f32>, filterParams1: vec4<f32>
) -> vec4<f32> {
    let lineFrom = filterParams0.xy;
    let lineVector = filterParams0.zw;
    let radii = filterParams1.xy;
    let uvOrigin = filterParams1.zw;

    let dP = colorTexCoord - lineFrom;
    let dC = lineVector;
    let dR = radii.y - radii.x;

    let a = dot(dC, dC) - dR * dR;
    let b = dot(dP, dC) + radii.x * dR;
    let c = dot(dP, dP) - radii.x * radii.x;
    let discrim = b * b - a * c;

    var color = vec4<f32>(0.0);
    if (discrim != 0.0) {
        var ts = (sqrt(discrim) * vec2<f32>(1.0, -1.0) + vec2<f32>(b)) / vec2<f32>(a);
        if (ts.x > ts.y) {
            ts = ts.yx;
        }
        let t = select(ts.y, ts.x, ts.x >= 0.0);
        color = textureSample(colorTexture, smp, uvOrigin + vec2<f32>(t, 0.0));
    }

    return color;
}

fn filterBlur(
    colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>, colorTextureSize: vec2<f32>,
    filterParams0: vec4<f32>, filterParams1: vec4<f32>
) -> vec4<f32> {
    let srcOffsetScale = filterParams0.xy / colorTextureSize;
    let support = i32(filterParams0.z);
    var gaussCoeff = filterParams1.xyz;

    var gaussSum = gaussCoeff.x; // weight[0]

    var color = textureSample(colorTexture, smp, colorTexCoord) * gaussCoeff.x;
    gaussCoeff = vec3<f32>(gaussCoeff.xy * gaussCoeff.yz, gaussCoeff.z);

    for (var i = 1; i <= support; i += 2) {
        var gaussPartialSum = gaussCoeff.x;
        gaussCoeff = vec3<f32>(gaussCoeff.xy * gaussCoeff.yz, gaussCoeff.z);
        gaussPartialSum += gaussCoeff.x;

        let srcOffset = srcOffsetScale * (f32(i) + gaussCoeff.x / gaussPartialSum);
        color += (textureSample(colorTexture, smp, colorTexCoord - srcOffset) +
                  textureSample(colorTexture, smp, colorTexCoord + srcOffset)) * gaussPartialSum;
        gaussSum += 2.0 * gaussPartialSum;
        gaussCoeff = vec3<f32>(gaussCoeff.xy * gaussCoeff.yz, gaussCoeff.z);
    }

    return color / gaussSum;
}

fn filterColorMatrix(
    colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>,
    filterParams0: vec4<f32>, filterParams1: vec4<f32>, filterParams2: vec4<f32>, filterParams3: vec4<f32>, filterParams4: vec4<f32>
) -> vec4<f32> {
    let srcColor = textureSample(colorTexture, smp, colorTexCoord);
    let colorMatrix = mat4x4<f32>(filterParams0, filterParams1, filterParams2, filterParams3);
    return colorMatrix * srcColor + filterParams4;
}

fn filterNone(colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>) -> vec4<f32> {
    return textureSample(colorTexture, smp, colorTexCoord);
}

fn filterColor(
    colorTexCoord: vec2<f32>, colorTexture: texture_2d<f32>, gammaLUT: texture_2d<f32>, colorTextureSize: vec2<f32>,
    filterParams0: vec4<f32>, filterParams1: vec4<f32>, filterParams2: vec4<f32>, filterParams3: vec4<f32>, filterParams4: vec4<f32>,
    colorFilter: i32
) -> vec4<f32> {
    switch (colorFilter) {
        case COMBINER_CTRL_FILTER_RADIAL_GRADIENT: {
            return filterRadialGradient(colorTexCoord, colorTexture, filterParams0, filterParams1);
        }
        case COMBINER_CTRL_FILTER_BLUR: {
            return filterBlur(colorTexCoord, colorTexture, colorTextureSize, filterParams0, filterParams1);
        }
        case COMBINER_CTRL_FILTER_TEXT: {
            return filterText(colorTexCoord, colorTexture, gammaLUT, colorTextureSize, filterParams0, filterParams1, filterParams2);
        }
        case COMBINER_CTRL_FILTER_COLOR_MATRIX: {
            return filterColorMatrix(colorTexCoord, colorTexture, filterParams0, filterParams1, filterParams2, filterParams3, filterParams4);
        }
        default: {}
    }
    return filterNone(colorTexCoord, colorTexture);
}

// Compositing
fn compositeDivide(num: f32, denom: f32) -> f32 {
    return select(0.0, num / denom, denom != 0.0);
}

fn compositeColorDodge(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    var result: vec3<f32>;
    for(var i = 0; i < 3; i = i + 1) {
        if (destColor[i] == 0.0) {
            result[i] = 0.0;
        } else if (srcColor[i] == 1.0) {
            result[i] = 1.0;
        } else {
            result[i] = destColor[i] / (1.0 - srcColor[i]);
        }
    }
    return result;
}

// https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB_alternative
fn compositeHSLToRGB(hsl: vec3<f32>) -> vec3<f32> {
    let a = hsl.y * min(hsl.z, 1.0 - hsl.z);
    let ks = (vec3<f32>(0.0, 8.0, 4.0) + vec3<f32>(hsl.x * FRAC_6_PI)) % vec3<f32>(12.0);
    let k3 = ks - vec3<f32>(3.0);
    let k9 = vec3<f32>(9.0) - ks;
    let minK = vec3<f32>(min(k3.x, k9.x), min(k3.y, k9.y), min(k3.z, k9.z));
    return vec3<f32>(hsl.z) - clamp(minK, vec3<f32>(-1.0), vec3<f32>(1.0)) * a;
}

// https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
fn compositeRGBToHSL(rgb: vec3<f32>) -> vec3<f32> {
    let v = max(max(rgb.r, rgb.g), rgb.b);
    let xMin = min(min(rgb.r, rgb.g), rgb.b);
    let c = v - xMin;
    let l = mix(xMin, v, 0.5);

    var terms: vec3<f32>;
    if (rgb.r == v) {
        terms = vec3<f32>(0.0, rgb.g, rgb.b);
    } else if (rgb.g == v) {
        terms = vec3<f32>(2.0, rgb.b, rgb.r);
    } else {
        terms = vec3<f32>(4.0, rgb.r, rgb.g);
    }

    let h = FRAC_PI_3 * compositeDivide(terms.x * c + terms.y - terms.z, c);
    let s = compositeDivide(c, v);
    return vec3<f32>(h, s, l);
}

fn compositeScreen(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    return destColor + srcColor - destColor * srcColor;
}

fn compositeHardLight(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(compositeScreen(destColor, vec3<f32>(2.0) * srcColor - vec3<f32>(1.0)).x, destColor.x * 2.0 * srcColor.x, srcColor.x <= 0.5),
        select(compositeScreen(destColor, vec3<f32>(2.0) * srcColor - vec3<f32>(1.0)).y, destColor.y * 2.0 * srcColor.y, srcColor.y <= 0.5),
        select(compositeScreen(destColor, vec3<f32>(2.0) * srcColor - vec3<f32>(1.0)).z, destColor.z * 2.0 * srcColor.z, srcColor.z <= 0.5)
    );
}

fn compositeSoftLight(destColor: vec3<f32>, srcColor: vec3<f32>) -> vec3<f32> {
    var darkenedDestColor: vec3<f32>;
    for (var i = 0; i < 3; i = i + 1) {
        darkenedDestColor[i] = select(sqrt(destColor[i]), ((16.0 * destColor[i] - 12.0) * destColor[i] + 4.0) * destColor[i], destColor[i] <= 0.25);
    }
    var factor: vec3<f32>;
    for (var i = 0; i < 3; i = i + 1) {
        factor[i] = select(darkenedDestColor[i] - destColor[i], destColor[i] * (1.0 - destColor[i]), srcColor[i] <= 0.5);
    }
    return destColor + (srcColor * 2.0 - 1.0) * factor;
}

fn compositeHSL(destColor: vec3<f32>, srcColor: vec3<f32>, op: i32) -> vec3<f32> {
    switch (op) {
        case COMBINER_CTRL_COMPOSITE_HUE: {
            return vec3<f32>(srcColor.x,  destColor.y, destColor.z);
        }
        case COMBINER_CTRL_COMPOSITE_SATURATION: {
            return vec3<f32>(destColor.x, srcColor.y,  destColor.z);
        }
        case COMBINER_CTRL_COMPOSITE_COLOR: {
            return vec3<f32>(srcColor.x,  srcColor.y,  destColor.z);
        }
        default: {
            return vec3<f32>(destColor.x, destColor.y, srcColor.z);
        }
    }
}

fn compositeRGB(destColor: vec3<f32>, srcColor: vec3<f32>, op: i32) -> vec3<f32> {
    switch (op) {
        case COMBINER_CTRL_COMPOSITE_MULTIPLY: {
            return destColor * srcColor;
        }
        case COMBINER_CTRL_COMPOSITE_SCREEN: {
            return compositeScreen(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_OVERLAY: {
            return compositeHardLight(srcColor, destColor);
        }
        case COMBINER_CTRL_COMPOSITE_DARKEN: {
            return min(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_LIGHTEN: {
            return max(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_COLOR_DODGE: {
            return compositeColorDodge(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_COLOR_BURN: {
            return vec3<f32>(1.0) - compositeColorDodge(vec3<f32>(1.0) - destColor, vec3<f32>(1.0) - srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_HARD_LIGHT: {
            return compositeHardLight(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_SOFT_LIGHT: {
            return compositeSoftLight(destColor, srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_DIFFERENCE: {
            return abs(destColor - srcColor);
        }
        case COMBINER_CTRL_COMPOSITE_EXCLUSION: {
            return destColor + srcColor - vec3<f32>(2.0) * destColor * srcColor;
        }
        case COMBINER_CTRL_COMPOSITE_HUE, COMBINER_CTRL_COMPOSITE_SATURATION, COMBINER_CTRL_COMPOSITE_COLOR, COMBINER_CTRL_COMPOSITE_LUMINOSITY: {
            return compositeHSLToRGB(compositeHSL(compositeRGBToHSL(destColor), compositeRGBToHSL(srcColor), op));
        }
        default: {
            return srcColor;
        }
    }
}

fn composite(srcColor: vec4<f32>, destTexture: texture_2d<f32>, destTextureSize: vec2<f32>, fragCoord: vec2<f32>, op: i32) -> vec4<f32> {
    if (op == COMBINER_CTRL_COMPOSITE_NORMAL) {
        return srcColor;
    }
    // FIXME(pcwalton): What should the output alpha be here?
    let destTexCoord = fragCoord / destTextureSize;
    let destColor = textureSample(destTexture, smp, destTexCoord);
    let blendedRGB = compositeRGB(destColor.rgb, srcColor.rgb, op);
    return vec4<f32>(
        srcColor.a * (1.0 - destColor.a) * srcColor.rgb + srcColor.a * destColor.a * blendedRGB + (1.0 - srcColor.a) * destColor.rgb,
        1.0
    );
}

// Masks
fn sampleMask(maskAlpha: f32, maskTexture: texture_2d<f32>, maskTextureSize: vec2<f32>, maskTexCoord: vec3<f32>, maskCtrl: i32) -> f32 {
    if (maskCtrl == 0) { return maskAlpha; }
    let maskTexCoordI = vec2<i32>(floor(maskTexCoord.xy));
    let texel = textureSample(maskTexture, smp, (vec2<f32>(maskTexCoordI / vec2<i32>(1, 4)) + 0.5) / maskTextureSize);

    var coverage = texel[maskTexCoordI.y % 4] + maskTexCoord.z;
    if ((maskCtrl & TILE_CTRL_MASK_WINDING) != 0) {
        coverage = abs(coverage);
    } else {
        coverage = 1.0 - abs(1.0 - (coverage - 2.0 * floor(coverage / 2.0)));
    }

    return min(maskAlpha, coverage);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let fragCoord = input.position;

    let ctrl = i32(input.vCtrl);
    let tileCtrl = i32(input.vTileCtrl);

    // Sample alpha from the mask texture.
    let maskCtrl0 = (tileCtrl >> TILE_CTRL_MASK_0_SHIFT) & TILE_CTRL_MASK_MASK;
    var maskAlpha = 1.0;
    maskAlpha = sampleMask(maskAlpha, uMaskTexture0, globals.uMaskTextureSize0, input.vMaskTexCoord0, maskCtrl0);

    // Get base color.
    var color = input.vBaseColor;

    // Get color combine flag.
    let color0Combine = (ctrl >> COMBINER_CTRL_COLOR_COMBINE_SHIFT) & COMBINER_CTRL_COLOR_COMBINE_MASK;

    // Do combining.
    if (color0Combine != 0) {
        // Get color filter flag.
        let color0Filter = (ctrl >> COMBINER_CTRL_COLOR_FILTER_SHIFT) & COMBINER_CTRL_FILTER_MASK;

        // Do filtering.
        let color0 = filterColor(
            input.vColorTexCoord0, uColorTexture0, uGammaLUT, globals.uColorTextureSize0,
            input.vFilterParams0, input.vFilterParams1, input.vFilterParams2, input.vFilterParams3, input.vFilterParams4,
            color0Filter
        );
        color = combineColor0(color, color0, color0Combine);
    }

    // Apply mask alpha.
    color.a *= maskAlpha;

    // Apply composite.
    let compositeOp = (ctrl >> COMBINER_CTRL_COMPOSITE_SHIFT) & COMBINER_CTRL_COMPOSITE_MASK;
    color = composite(color, uDestTexture, globals.uFramebufferSize, fragCoord.xy, compositeOp);

    // Premultiply alpha.
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}