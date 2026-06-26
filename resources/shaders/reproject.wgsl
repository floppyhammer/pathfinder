// pathfinder/resources/shaders/reproject.wgsl

struct Globals {
    uNewTransform: mat4x4<f32>,
    uOldTransform: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var uTexture: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VertexInput {
    @location(0) aPosition: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vTexCoord: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos_f = vec2<f32>(input.aPosition);
    out.vTexCoord = pos_f;

    let y = 1.0 - pos_f.y;
    let pos = globals.uNewTransform * vec4<f32>(pos_f.x, y, 0.0, 1.0);
    out.position = vec4<f32>(pos.x, -pos.y, pos.z, pos.w);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normTexCoord = globals.uOldTransform * vec4<f32>(input.vTexCoord, 0.0, 1.0);
    let texCoord = ((normTexCoord.xy / normTexCoord.w) + 1.0) * 0.5;
    return textureSample(uTexture, smp, texCoord);
}
