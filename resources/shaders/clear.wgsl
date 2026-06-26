// pathfinder/resources/shaders/clear.wgsl

struct Globals {
    uRect: vec4<f32>,
    uFramebufferSize: vec2<f32>,
    uColor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) aPosition: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos_f = vec2<f32>(input.aPosition);
    let position = mix(globals.uRect.xy, globals.uRect.zw, pos_f) / globals.uFramebufferSize * 2.0 - 1.0;
    out.position = vec4<f32>(position.x, -position.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(globals.uColor.rgb, 1.0) * globals.uColor.a;
}
