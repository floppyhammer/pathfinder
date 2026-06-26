// pathfinder/resources/shaders/d3d11/bin.wgsl
// Assigns microlines to tiles. Generate fills used in fill.comp.

struct Uniforms {
    uMicrolineCount: i32,
    uMaxFillCount: i32, // How many slots we have allocated for fills.
    uPad0: i32,
    uPad1: i32,
};
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

struct Microlines {
    data: array<vec4<u32>>,
};
@group(1) @binding(0) var<storage, read> bMicrolines: Microlines;

struct Metadata {
    // [0]: tile rect
    // [1].x: tile offset
    // [1].y: path ID
    // [1].z: z write flag
    // [1].w: clip path ID
    // [2].x: backdrop offset
    data: array<vec4<i32>>,
};
@group(1) @binding(1) var<storage, read> bMetadata: Metadata;

struct IndirectDrawParams {
    // [0]: vertexCount (6)
    // [1]: instanceCount (of fills)
    // [2]: vertexStart (0)
    // [3]: baseInstance (0)
    // [4]: alpha tile count
    data: array<atomic<u32>>,
};
@group(1) @binding(2) var<storage, read_write> bIndirectDrawParams: IndirectDrawParams;

struct Fills {
    data: array<u32>,
};
@group(1) @binding(3) var<storage, read_write> bFills: Fills;

struct Tiles {
    // [0]: next tile ID (initialized to -1)
    // [1]: first fill ID (initialized to -1)
    // [2]: backdrop delta upper 8 bits, alpha tile ID lower 24 (initialized to 0, -1 respectively)
    // [3]: color/ctrl/backdrop word
    data: array<atomic<u32>>,
};
@group(1) @binding(4) var<storage, read_write> bTiles: Tiles;

struct Backdrops {
    // [0]: backdrop
    // [1]: tile X offset
    // [2]: path ID
    data: array<atomic<u32>>,
};
@group(1) @binding(5) var<storage, read_write> bBackdrops: Backdrops;

const MAX_ITERATIONS: u32 = 1024u;

const STEP_DIRECTION_NONE: i32 = 0;
const STEP_DIRECTION_X: i32 = 1;
const STEP_DIRECTION_Y: i32 = 2;

const TILE_FIELD_NEXT_TILE_ID: u32 = 0u;
const TILE_FIELD_FIRST_FILL_ID: u32 = 1u;
const TILE_FIELD_BACKDROP_ALPHA_TILE_ID: u32 = 2u;
const TILE_FIELD_CONTROL: u32 = 3u;

fn computeTileIndexNoCheck(tileCoords: vec2<i32>, pathTileRect: vec4<i32>, pathTileOffset: u32) -> u32 {
    let offsetCoords = tileCoords - pathTileRect.xy;
    return u32(i32(pathTileOffset) + offsetCoords.x + offsetCoords.y * (pathTileRect.z - pathTileRect.x));
}

fn computeTileOutcodes(tileCoords: vec2<i32>, pathTileRect: vec4<i32>) -> vec4<bool> {
    return vec4<bool>(
        tileCoords.x < pathTileRect.x,
        tileCoords.y < pathTileRect.y,
        tileCoords.x >= pathTileRect.z,
        tileCoords.y >= pathTileRect.w
    );
}

fn computeTileIndex(
    tileCoords: vec2<i32>,
    pathTileRect: vec4<i32>,
    pathTileOffset: u32,
    outTileIndex: ptr<function, u32>
) -> bool {
    *outTileIndex = computeTileIndexNoCheck(tileCoords, pathTileRect, pathTileOffset);
    return !any(computeTileOutcodes(tileCoords, pathTileRect));
}

fn addFill(lineSegment: vec4<f32>, tileCoords: vec2<i32>, pathTileRect: vec4<i32>, pathTileOffset: u32) {
    // Compute tile offset. If out of bounds, cull.
    var tileIndex: u32 = 0u;
    if (!computeTileIndex(tileCoords, pathTileRect, pathTileOffset, &tileIndex)) {
        return;
    }

    // Clip line. If too narrow, cull.
    let scaledLocalLine = vec4<u32>(vec4<i32>(round((lineSegment - vec4<f32>(tileCoords.xyxy * vec4<i32>(16).xyxy)) * 256.0)));
    if (scaledLocalLine.x == scaledLocalLine.z) {
        return;
    }

    // Bump instance count.
    let fillIndex = atomicAdd(&bIndirectDrawParams.data[1], 1u);

    // Fill out the link field, inserting into the linked list.
    let fillLink = atomicExchange(&bTiles.data[tileIndex * 4u + TILE_FIELD_FIRST_FILL_ID], fillIndex);

    // Write fill.
    if (fillIndex < u32(uniforms.uMaxFillCount)) {
        bFills.data[fillIndex * 3u + 0u] = (scaledLocalLine.x & 0xffffu) | (scaledLocalLine.y << 16u);
        bFills.data[fillIndex * 3u + 1u] = (scaledLocalLine.z & 0xffffu) | (scaledLocalLine.w << 16u);
        bFills.data[fillIndex * 3u + 2u] = fillLink;
    }
}

fn adjustBackdrop(
    backdropDelta: i32,
    tileCoords: vec2<i32>,
    pathTileRect: vec4<i32>,
    pathTileOffset: u32,
    pathBackdropOffset: u32
) {
    let outcodes = computeTileOutcodes(tileCoords, pathTileRect);
    if (any(outcodes)) {
        if (!outcodes.x && outcodes.y && !outcodes.z) {
            let backdropIndex = pathBackdropOffset + u32(tileCoords.x - pathTileRect.x);
            atomicAdd(&bBackdrops.data[backdropIndex * 3u], u32(backdropDelta));
        }
    } else {
        let tileIndex = computeTileIndexNoCheck(tileCoords, pathTileRect, pathTileOffset);
        atomicAdd(&bTiles.data[tileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID], u32(backdropDelta) << 24u);
    }
}

fn unpackMicroline(segmentIndex: u32, outPathIndex: ptr<function, u32>) -> vec4<f32> {
    *outPathIndex = bMicrolines.data[segmentIndex].w;
    let signedMicroline = vec4<i32>(bMicrolines.data[segmentIndex]);

    return vec4<f32>(
        f32((signedMicroline.x << 16u) >> 16u),
        f32(signedMicroline.x >> 16u),
        f32((signedMicroline.y << 16u) >> 16u),
        f32(signedMicroline.y >> 16u)
    ) + vec4<f32>(
        f32(signedMicroline.z & 0xff),
        f32((signedMicroline.z >> 8u) & 0xff),
        f32((signedMicroline.z >> 16u) & 0xff),
        f32((signedMicroline.z >> 24u) & 0xff)
    ) / 256.0;
}

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let segmentIndex = global_id.x;
    if (segmentIndex >= u32(uniforms.uMicrolineCount)) {
        return;
    }

    var pathIndex: u32 = 0u;
    let lineSegment = unpackMicroline(segmentIndex, &pathIndex);

    let pathTileRect = bMetadata.data[pathIndex * 3u + 0u];
    let pathTileOffset = u32(bMetadata.data[pathIndex * 3u + 1u].x);
    let pathBackdropOffset = u32(bMetadata.data[pathIndex * 3u + 2u].x);

    // Following is a straight port of `process_line_segment()`:
    let tileSize = vec2<i32>(16, 16);
    let tileLineSegment = vec4<i32>(floor(lineSegment / vec4<f32>(16.0)));
    let fromTileCoords = tileLineSegment.xy;
    let toTileCoords = tileLineSegment.zw;

    let vector_ = lineSegment.zw - lineSegment.xy;
    var tileStep = vec2<i32>(1, 1);
    if (vector_.x < 0.0) { tileStep.x = -1; }
    if (vector_.y < 0.0) { tileStep.y = -1; }

    var xCrossingShift: i32 = 1; if (vector_.x < 0.0) { xCrossingShift = 0; }
    var yCrossingShift: i32 = 1; if (vector_.y < 0.0) { yCrossingShift = 0; }

    let firstTileCrossing = vec2<f32>(vec2<i32>(fromTileCoords + vec2<i32>(xCrossingShift, yCrossingShift)) * tileSize);
    var tMax = (firstTileCrossing - lineSegment.xy) / vector_;
    let tDelta = abs(vec2<f32>(tileSize) / vector_);

    var currentPosition = lineSegment.xy;
    var tileCoords = fromTileCoords;
    var lastStepDirection = STEP_DIRECTION_NONE;
    var iteration = 0u;

    while (iteration < MAX_ITERATIONS) {
        var nextStepDirection: i32;
        if (tMax.x < tMax.y) {
            nextStepDirection = STEP_DIRECTION_X;
        } else if (tMax.x > tMax.y) {
            nextStepDirection = STEP_DIRECTION_Y;
        } else if (tileStep.x > 0) {
            nextStepDirection = STEP_DIRECTION_X;
        } else {
            nextStepDirection = STEP_DIRECTION_Y;
        }

        var nextT = min(tMax.y, 1.0);
        if (nextStepDirection == STEP_DIRECTION_X) {
            nextT = min(tMax.x, 1.0);
        }

        // If we've reached the end tile, don't step at all.
        if (all(tileCoords == toTileCoords)) {
            nextStepDirection = STEP_DIRECTION_NONE;
        }

        let nextPosition = mix(lineSegment.xy, lineSegment.zw, vec2<f32>(nextT));

        let clippedLineSegment = vec4<f32>(currentPosition, nextPosition);
        addFill(clippedLineSegment, tileCoords, pathTileRect, pathTileOffset);

        // Add extra fills if necessary.
        var auxiliarySegment: vec4<f32>;
        var haveAuxiliarySegment = false;
        if (tileStep.y < 0 && nextStepDirection == STEP_DIRECTION_Y) {
            auxiliarySegment = vec4<f32>(clippedLineSegment.zw, vec2<f32>(tileCoords * tileSize));
            haveAuxiliarySegment = true;
        } else if (tileStep.y > 0 && lastStepDirection == STEP_DIRECTION_Y) {
            auxiliarySegment = vec4<f32>(vec2<f32>(tileCoords * tileSize), clippedLineSegment.xy);
            haveAuxiliarySegment = true;
        }
        if (haveAuxiliarySegment) {
            addFill(auxiliarySegment, tileCoords, pathTileRect, pathTileOffset);
        }

        // Adjust backdrop if necessary.
        //
        // NB: Do not refactor the calls below. This exact code sequence is needed to avoid a
        // miscompilation on the Radeon Metal compiler.
        if (tileStep.x < 0 && lastStepDirection == STEP_DIRECTION_X) {
            adjustBackdrop(1, tileCoords, pathTileRect, pathTileOffset, pathBackdropOffset);
        } else if (tileStep.x > 0 && nextStepDirection == STEP_DIRECTION_X) {
            adjustBackdrop(-1, tileCoords, pathTileRect, pathTileOffset, pathBackdropOffset);
        }

        // Take a step.
        if (nextStepDirection == STEP_DIRECTION_X) {
            tMax.x += tDelta.x;
            tileCoords.x += tileStep.x;
        } else if (nextStepDirection == STEP_DIRECTION_Y) {
            tMax.y += tDelta.y;
            tileCoords.y += tileStep.y;
        } else {
            break;
        }

        currentPosition = nextPosition;
        lastStepDirection = nextStepDirection;
        iteration++;
    }
}