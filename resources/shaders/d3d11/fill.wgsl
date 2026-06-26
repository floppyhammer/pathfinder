// pathfinder/resources/shaders/d3d11/fill.wgsl

struct Globals {
    uAlphaTileRange: vec2<i32>,
    uPad0: vec2<i32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var uDest: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(2) var uAreaLUT: texture_2d<f32>;
@group(0) @binding(3) var uAreaLUTSampler: sampler;

struct Fills {
    data: array<u32>,
};
@group(1) @binding(0) var<storage, read> bFills: Fills;

struct Tiles {
    data: array<u32>,
};
@group(1) @binding(1) var<storage, read> bTiles: Tiles;

struct AlphaTiles {
    data: array<u32>,
};
@group(1) @binding(2) var<storage, read> bAlphaTiles: AlphaTiles;

const TILE_FIELD_NEXT_TILE_ID: u32 = 0u;
const TILE_FIELD_FIRST_FILL_ID: u32 = 1u;
const TILE_FIELD_BACKDROP_ALPHA_TILE_ID: u32 = 2u;
const TILE_FIELD_CONTROL: u32 = 3u;

const TILE_CTRL_MASK_MASK: u32 = 0x3u;
const TILE_CTRL_MASK_WINDING: u32 = 0x1u;
const TILE_CTRL_MASK_EVEN_ODD: u32 = 0x2u;
const TILE_CTRL_MASK_0_SHIFT: u32 = 0u;

fn computeCoverage(from_: vec2<f32>, to_: vec2<f32>) -> vec4<f32> {
    // Determine winding, and sort into a consistent order so we only need to find one root below.
    var left = to_;
    var right = from_;
    if (from_.x < to_.x) {
        left = from_;
        right = to_;
    }

    // Shoot a vertical ray toward the curve.
    let window = clamp(vec2<f32>(from_.x, to_.x), vec2<f32>(-0.5), vec2<f32>(0.5));
    let offset = mix(window.x, window.y, 0.5) - left.x;

    // On-segment coordinate.
    let t = offset / (right.x - left.x);

    // Compute position and derivative to form a line approximation.
    let y = mix(left.y, right.y, t);
    // CHY: y position calculated from t.
    let d = (right.y - left.y) / (right.x - left.x);
    // CHY: Derivative of the segment.

    // Look up area under that line, and scale horizontally to the window size.
    let dX = window.x - window.y;

    // Return the color at the specific position in texture areaLUT.
    let uv = vec2<f32>(y + 8.0, abs(d * dX)) / 16.0;
    return textureSampleLevel(uAreaLUT, uAreaLUTSampler, uv, 0.0) * dX;
}

fn accumulateCoverageForFillList(firstFillIndex: i32, tileSubCoord: vec2<i32>) -> vec4<f32> {
    let tileFragCoord = vec2<f32>(tileSubCoord) + vec2<f32>(0.5);
    // This might be the coverage mask.
    var coverages = vec4<f32>(0.0);

    var fillIndex = firstFillIndex;
    var iteration = 0;

    loop {
        if (!(fillIndex >= 0 && iteration < 1024)) {
            break;
        }

        // iFills[fillFrom, fillTo, ?, fillFrom, fillTo, ?, ...]
        // What is the third element?
        let fillFrom = bFills.data[u32(fillIndex) * 3u + 0u];
        let fillTo   = bFills.data[u32(fillIndex) * 3u + 1u];

        // Pack: lineSegment = vec4(from.x, from.y, to.x, to.y).
        let lineSegment = vec4<f32>(
            f32(fillFrom & 0xffffu),
            f32(fillFrom >> 16u),
            f32(fillTo & 0xffffu),
            f32(fillTo >> 16u)
        ) / 256.0 - tileFragCoord.xyxy;

        // Compute if this texel is covered by the fill?
        coverages += computeCoverage(lineSegment.xy, lineSegment.zw);

        fillIndex = i32(bFills.data[u32(fillIndex) * 3u + 2u]);
        iteration++;
    }

    return coverages;
}

fn computeTileCoord(alphaTileIndex: u32, localInvocationID: vec3<u32>) -> vec2<i32> {
    let x = alphaTileIndex & 0xffu;
    let y = ((alphaTileIndex >> 8u) & 0xffu) | (((alphaTileIndex >> 16u) & 0xffu) << 8u);
    let coords = vec2<u32>(16u, 4u) * vec2<u32>(x, y) + localInvocationID.xy;
    return vec2<i32>(coords);
}

@compute @workgroup_size(16, 4)
fn cs_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    // Local coordinates out of local size (16, 4) * ivec2(1, 4).
    let tileSubCoord = vec2<i32>(local_id.xy) * vec2<i32>(1, 4);

    // This is a workaround for the 64K workgroup dispatch limit in OpenGL.
    let batchAlphaTileIndex = group_id.x | (group_id.y << 15u);

    let alphaTileIndex = batchAlphaTileIndex + u32(globals.uAlphaTileRange.x);
    if (alphaTileIndex >= u32(globals.uAlphaTileRange.y)) {
        return;
    }

    let tileIndex = bAlphaTiles.data[batchAlphaTileIndex * 2u + 0u];

    // |?(8bit)|x(24bit)| -> |0(8bit)|x(24bit)|
    // Commented to fix artifacts in OpenGL.
    // let backdropAlphaTileField = bTiles.data[tileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID];
    // if ((i32(backdropAlphaTileField << 8u) >> 8u) < 0) {
    //     return;
    // }

    let fillIndex = i32(bTiles.data[tileIndex * 4u + TILE_FIELD_FIRST_FILL_ID]);
    let tileControlWord = bTiles.data[tileIndex * 4u + TILE_FIELD_CONTROL];
    let backdrop = i32(tileControlWord) >> 24u;

    var coverages = vec4<f32>(f32(backdrop));
    coverages += accumulateCoverageForFillList(fillIndex, tileSubCoord);

    let tileCtrl = i32((tileControlWord >> 16u) & 0xffu);
    let maskCtrl = (tileCtrl >> TILE_CTRL_MASK_0_SHIFT) & i32(TILE_CTRL_MASK_MASK);

    if ((u32(maskCtrl) & TILE_CTRL_MASK_WINDING) != 0u) {
        coverages = clamp(abs(coverages), vec4<f32>(0.0), vec4<f32>(1.0));
    } else {
        // WGSL 没自带 mod 函数，使用标准的 x - y * floor(x / y) 代替 GLSL 的 mod(coverages, 2.0)
        let modResult = coverages - 2.0 * floor(coverages / 2.0);
        coverages = clamp(1.0 - abs(1.0 - modResult), vec4<f32>(0.0), vec4<f32>(1.0));
    }

    // Handle clip if necessary.
    // clipTileIndex should be converted to int first, as it might be negative.
    let clipTileIndex = i32(bAlphaTiles.data[batchAlphaTileIndex * 2u + 1u]);
    if (clipTileIndex >= 0) {
        let clipCoords = computeTileCoord(u32(clipTileIndex), local_id);
        coverages = min(coverages, textureLoad(uDest, clipCoords));
    }

    let storeCoords = computeTileCoord(alphaTileIndex, local_id);
    textureStore(uDest, storeCoords, coverages);
}