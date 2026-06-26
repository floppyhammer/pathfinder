// pathfinder/resources/shaders/demo_ground.wgsl

struct Globals {
    uTransform: mat4x4<f32>,
    uGridlineCount: f32,
    uGroundColor: vec4<f32>,
    uGridlineColor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) aPosition: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vPosition: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos_f = vec2<f32>(input.aPosition);
    out.vPosition = pos_f;
    let pos = globals.uTransform * vec4<f32>(pos_f, 0.0, 1.0);
    out.position = vec4<f32>(pos.x, -pos.y, pos.z, pos.w);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let grid_pos = input.vPosition * globals.uGridlineCount;
    let cell = floor(grid_pos);
    let t = grid_pos - cell;
    
    let line_thickness = 1.0 / globals.uGridlineCount;
    let is_line = (t.x < line_thickness) || (t.y < line_thickness);
    
    return mix(globals.uGroundColor, globals.uGridlineColor, f32(is_line));
}
