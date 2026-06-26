// pathfinder/resources/shaders/d3d9/tile_clip_copy.wgsl
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
    @location(1) aTileIndex: i32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vTexCoord: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Unpack grid position from 1D tile index.
    let gridPos = vec2<i32>(input.aTileIndex % 256, input.aTileIndex / 256);
    var position = vec2<f32>(gridPos + vec2<i32>(input.aTileOffset));

    // Scale to normalized UV space coordinates.
    position *= vec2<f32>(16.0, 4.0) / globals.uFramebufferSize;

    out.vTexCoord = position;

    // Handle culled or invalid tiles.
    if (input.aTileIndex < 0) {
        position = vec2<f32>(0.0);
    }

    // PF_ORIGIN_UPPER_LEFT equivalent: Invert Y-axis for WebGPU clip space requirements.
    let y = 1.0 - position.y;
    let globalPos = mix(vec2<f32>(-1.0), vec2<f32>(1.0), vec2<f32>(position.x, y));
    out.position = vec4<f32>(globalPos, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Directly sample from the source mask texture.
    return textureSample(uSrc, smp, input.vTexCoord);
}