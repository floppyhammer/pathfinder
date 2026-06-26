// pathfinder/resources/shaders/d3d9/fill.wgsl
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Globals {
    tile_size: vec2<f32>, // Fixed as (16, 16).
    framebuffer_size: vec2<f32>, // Mask framebuffer. Dynamic as (4096, 1024 * page_count).
};

@group(0) @binding(0) var<uniform> globals: Globals;
// Pre-prepared texture "area-lut.png" of size (256, 256).
@group(0) @binding(1) var areaLUT: texture_2d<f32>;
@group(0) @binding(2) var areaLUTSampler: sampler;

struct VertexInput {
    @location(0) TessCoord: vec2<u32>, // Vertex coordinates in a quad, fixed.
    @location(1) LineSegment: vec4<u32>, // Line segment from the built batch.
    @location(2) TileIndex: u32, // Alpha tile index.
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) from_: vec2<f32>,
    @location(1) to_: vec2<f32>,
};

/// Tile index -> index coordinates -> pixel coordinates.
fn computeTileOffset(tileIndex: u32, framebufferSize: vec2<f32>, tileSize: vec2<f32>) -> vec2<f32> {
    // Tiles count per row in the mask texture.
    let tilesPerRow = u32(framebufferSize.x / tileSize.x);

    // Tile index coordinates.
    let tileX = f32(tileIndex % tilesPerRow);
    let tileY = f32(tileIndex / tilesPerRow);

    // Pixel coordinates of the tile's origin.
    // We compress data into RGBA channels in the vertical direction.
    // That's why we have vec2(1.0f, 0.25f) here.
    return vec2<f32>(tileX, tileY) * tileSize * vec2<f32>(1.0, 0.25);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let tileSize = globals.tile_size;
    let framebufferSize = globals.framebuffer_size;

    // Unpack endpoints.
    let from_ = vec2<f32>(f32(input.LineSegment.x) / 256.0, f32(input.LineSegment.y) / 256.0);
    let to_   = vec2<f32>(f32(input.LineSegment.z) / 256.0, f32(input.LineSegment.w) / 256.0);

    // Get the global origin of the tile.
    let tileOrigin = computeTileOffset(input.TileIndex, framebufferSize, tileSize);

    var position: vec2<f32>;

    // CORE STEP
    // Default square quad -> the fill quad encircled by the line segment, the bottom bound of the tile,
    // and two vertical auxiliary segments.
    if (input.TessCoord.x == 0u) {
        position.x = floor(min(from_.x, to_.x)); // Left
    } else {
        position.x = ceil(max(from_.x, to_.x)); // Right
    }

    if (input.TessCoord.y == 0u) {
        position.y = floor(min(from_.y, to_.y)); // Top
    } else {
        position.y = tileSize.y; // Bottom, which is always the bottom bound of the tile.
    }

    // Compress the fill quad in the vertical direction.
    position.y = floor(position.y * 0.25);

    // Since each fragment corresponds to 4 pixels on a scanline, the varying interpolation will
    // land the fragment halfway between the four-pixel strip, at pixel offset 2.0. But we want to
    // do our coverage calculation on the center of the first pixel in the strip instead, at pixel
    // offset 0.5. This adjustment of 1.5 accomplishes that.
    let offset = vec2<f32>(0.0, 1.5) - position * vec2<f32>(1.0, 4.0);
    output.from_ = from_ + offset;
    output.to_ = to_ + offset;

    // Global pixel position -> normalized UV position.
    let globalPosition = (tileOrigin + position) / framebufferSize;

    // Map [0, 1] to clip space [-1, 1]
    let clipPosition = globalPosition * 2.0 - 1.0;

    output.position = vec4<f32>(clipPosition.x, -clipPosition.y, 0.0, 1.0);
    return output;
}

// === Fragment Shader Helper Functions ===

/// Understanding this process is quite hard as we need to understand the areaLUT texture first.
/// But I guess areaLUT is mostly used for anti-aliasing.
fn computeCoverage(from_vec: vec2<f32>, to_vec: vec2<f32>, areaLUTTex: texture_2d<f32>, areaLUTSmp: sampler) -> vec4<f32> {
    // Determine winding, and sort into a consistent order so we only need to find one root below.
    var left: vec2<f32>;
    var right: vec2<f32>;
    if (from_vec.x < to_vec.x) {
        left = from_vec;
        right = to_vec;
    } else {
        left = to_vec;
        right = from_vec;
    }

    // Shoot a vertical ray toward the curve.
    let window = clamp(vec2<f32>(from_vec.x, to_vec.x), vec2<f32>(-0.5), vec2<f32>(0.5));
    let offset = mix(window.x, window.y, 0.5) - left.x;

    // On-segment coordinate.
    let t = offset / (right.x - left.x);

    // Compute position and derivative to form a line approximation.
    let y = mix(left.y, right.y, t); // Scanline hit position y calculated from t.
    let d = (right.y - left.y) / (right.x - left.x); // Derivative of the line segment.

    // Look up area under that line, and scale horizontally to the window size.
    let dX = window.x - window.y;

    // Look up coverage in the LUT.
    return textureSample(areaLUTTex, areaLUTSmp, vec2<f32>(y + 8.0, abs(d * dX)) / 16.0) * dX;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Compute the coverage vector (representing 4 horizontal pixels).
    let color = computeCoverage(input.from_, input.to_, areaLUT, areaLUTSampler);

    // Return the calculated coverage color directly to the mask framebuffer.
    return color;
}