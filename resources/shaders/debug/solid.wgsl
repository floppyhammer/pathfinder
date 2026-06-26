@group(0) @binding(0)
var<uniform> uTransform: mat4x4<f32>;

@group(0) @binding(1)
var<uniform> uColor: vec4<f32>;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return uTransform * vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uColor;
}
