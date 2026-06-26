@group(0) @binding(0)
var<uniform> uTransform: mat4x4<f32>;

@group(0) @binding(1)
var<uniform> uColor: vec4<f32>;

@group(0) @binding(2)
var uTextureSampler: sampler;

@group(0) @binding(3)
var uTexture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) tex_coord: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = uTransform * vec4<f32>(position, 0.0, 1.0);
    output.tex_coord = tex_coord;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var tex_value = textureSample(uTexture, uTextureSampler, input.tex_coord).x;
    return vec4<f32>(tex_value, tex_value, tex_value, tex_value) * uColor;
}
