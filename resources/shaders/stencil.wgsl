// pathfinder/resources/shaders/stencil.wgsl

struct Globals {
    uTransform: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) aPosition: vec2<u32>,
};

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    let pos_f = vec2<f32>(input.aPosition);
    let pos = globals.uTransform * vec4<f32>(pos_f, 0.0, 1.0);
    return vec4<f32>(pos.x, -pos.y, pos.z, pos.w);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // This should be color masked out.
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
