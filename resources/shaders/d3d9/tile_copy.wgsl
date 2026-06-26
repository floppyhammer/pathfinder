// pathfinder/resources/shaders/d3d9/tile_copy.wgsl

struct Globals {
    transform: mat4x4<f32>,
    tile_size: vec2<f32>,
    framebuffer_size: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var srcTexture: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VertexInput {
    @location(0) tile_position: vec2<i32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let position = vec2<f32>(input.tile_position) * globals.tile_size;
    let pos = globals.transform * vec4<f32>(position, 0.0, 1.0);
    out.position = vec4<f32>(pos.x, -pos.y, pos.z, pos.w);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let texCoord = frag_coord.xy / globals.framebuffer_size;
    return textureSample(srcTexture, smp, texCoord);
}
