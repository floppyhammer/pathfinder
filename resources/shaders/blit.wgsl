// pathfinder/resources/shaders/blit.wgsl

struct Globals {
    uDestRect: vec4<f32>,        // [x_min, y_min, x_max, y_max] in pixel coordinates
    uFramebufferSize: vec2<f32>, // [width, height] of the backend framebuffer
    uPad0: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var uSrc: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vTexCoord: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var out: VertexOutput;

    // 1. Generate standard big triangle UV and NDC coordinates covering the full screen (-1..1)
    // Vertex 0: UV(0, 0) -> NDC(-1,  1)
    // Vertex 1: UV(2, 0) -> NDC( 3,  1)
    // Vertex 2: UV(0, 2) -> NDC(-1, -3)
    let uv = vec2<f32>(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));

    // Save the original full-screen UV for fragment shader output
    out.vTexCoord = uv;

    // 2. Map the 0..1 full-screen space into the specified pixel rectangle (globals.uDestRect)
    let pixelPos = mix(globals.uDestRect.xy, globals.uDestRect.zw, uv);

    // 3. Map the pixel coordinates precisely into WebGPU NDC space (-1..1)
    // X axis: 0 -> -1.0, width -> 1.0
    // Y axis: 0 -> 1.0, height -> -1.0 (Note: WebGPU's Y axis points upwards, so we subtract from 1.0)
    let ndcX = (pixelPos.x / globals.uFramebufferSize.x) * 2.0 - 1.0;
    let ndcY = 1.0 - (pixelPos.y / globals.uFramebufferSize.y) * 2.0;

    out.position = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@location(0) vTexCoord: vec2<f32>) -> @location(0) vec4<f32> {
    // Discard fragments outside the 0..1 UV range to prevent artifacts
    // when the big triangle extends past the destination bounding box.
    if (vTexCoord.x > 1.0 || vTexCoord.y > 1.0) {
        discard;
    }
    return textureSample(uSrc, smp, vTexCoord);
}