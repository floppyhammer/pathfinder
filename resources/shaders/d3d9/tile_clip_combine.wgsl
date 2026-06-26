// pathfinder/resources/shaders/d3d9/tile_clip_combine.wgsl
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Globals {
    uTileSize: vec2<f32>,         // Fixed as (16, 16).
    uFramebufferSize: vec2<f32>,   // Mask framebuffer. Dynamic as (4096, 1024 * page_count).
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var uSrc: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VertexInput {
    @location(0) aTileOffset: vec2<u32>,
    @location(1) aDestTileIndex: i32,
    @location(2) aDestBackdrop: i32,
    @location(3) aSrcTileIndex: i32,
    @location(4) aSrcBackdrop: i32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vTexCoord0: vec2<f32>,
    @location(1) vBackdrop0: f32,
    @location(2) vTexCoord1: vec2<f32>,
    @location(3) vBackdrop1: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Unpack positions matching the original GLSL division/modulo step.
    let destGrid = vec2<i32>(input.aDestTileIndex % 256, input.aDestTileIndex / 256);
    let srcGrid  = vec2<i32>(input.aSrcTileIndex  % 256, input.aSrcTileIndex  / 256);

    var destPosition = vec2<f32>(destGrid + vec2<i32>(input.aTileOffset));
    var srcPosition  = vec2<f32>(srcGrid  + vec2<i32>(input.aTileOffset));

    // Scale to UV coordinates.
    destPosition *= vec2<f32>(16.0, 4.0) / globals.uFramebufferSize;
    srcPosition  *= vec2<f32>(16.0, 4.0) / globals.uFramebufferSize;

    out.vTexCoord0 = destPosition;
    out.vTexCoord1 = srcPosition;

    out.vBackdrop0 = f32(input.aDestBackdrop);
    out.vBackdrop1 = f32(input.aSrcBackdrop);

    if (input.aDestTileIndex < 0) {
        destPosition = vec2<f32>(0.0);
    }

    // Clip space transformation with WebGPU Y-axis flipping logic.
    let y = 1.0 - destPosition.y;
    let globalPos = mix(vec2<f32>(-1.0), vec2<f32>(1.0), vec2<f32>(destPosition.x, y));
    out.position = vec4<f32>(globalPos, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // In WebGPU, we fetch the red channel since mask buffers are usually R8/R16UNORM.
    let t0 = textureSample(uSrc, smp, input.vTexCoord0).r;
    let t1 = textureSample(uSrc, smp, input.vTexCoord1).r;

    // Combine mask values with their backdrops using min-abs operation.
    let res = min(abs(t0 + input.vBackdrop0), abs(t1 + input.vBackdrop1));

    return vec4<f32>(res, res, res, res);
}