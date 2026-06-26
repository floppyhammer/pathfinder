// pathfinder/resources/shaders/d3d11/dice.wgsl
// Chops lines and curves into microlines.

struct TransformUniform {
    uTransform: mat2x2<f32>,
    uTranslation: vec2<f32>,
    uPad0: vec2<f32>,
};

struct BatchUniform {
    uPathCount: i32,
    uLastBatchSegmentIndex: i32,
    uMaxMicrolineCount: i32,
    uPad1: i32,
};

@group(0) @binding(5) var<uniform> uUniform0: TransformUniform;
@group(0) @binding(6) var<uniform> uUniform1: BatchUniform;

struct ComputeIndirectParams {
    // [0]: number of x workgroups (actually not used)
    // [1]: number of y workgroups (always 1) (actually not used)
    // [2]: number of z workgroups (always 1) (actually not used)
    // [3]: number of output microlines
    iComputeIndirectParams: array<atomic<u32>>,
};
@group(1) @binding(0) var<storage, read_write> bComputeIndirectParams: ComputeIndirectParams;

struct DiceMetadata {
    // x: global path ID
    // y: first global segment index
    // z: first batch segment index
    // w: unused
    data: array<vec4<u32>>,
};
@group(1) @binding(1) var<storage, read> bDiceMetadata: DiceMetadata;

struct Points {
    data: array<vec2<f32>>,
};
@group(1) @binding(2) var<storage, read> bPoints: Points;

struct InputIndices {
    data: array<vec2<u32>>,
};
@group(1) @binding(3) var<storage, read> bInputIndices: InputIndices;

struct Microlines {
    // x: from (X, Y) whole pixels, packed signed 16-bit
    // y: to (X, Y) whole pixels, packed signed 16-bit
    // z: (from X, from Y, to X, to Y) fractional pixels, packed unsigned 8-bit (0.8 fixed point)
    // w: path ID
    data: array<vec4<u32>>,
};
@group(1) @binding(4) var<storage, read_write> bMicrolines: Microlines;

const BIN_WORKGROUP_SIZE: u32 = 64u;
const MAX_CURVE_STACK_SIZE: u32 = 32u;

const FLAGS_PATH_INDEX_CURVE_IS_QUADRATIC: u32 = 0x80000000u;
const FLAGS_PATH_INDEX_CURVE_IS_CUBIC: u32 = 0x40000000u;

const BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX: u32 = 3u;

const TOLERANCE: f32 = 0.25;
const MICROLINE_LENGTH: f32 = 16.0;

/// Save the obtained microline.
fn emitMicroline(microlineSegment: vec4<f32>, pathIndex: u32, outputMicrolineIndex: u32) {
    if (outputMicrolineIndex >= u32(uUniform1.uMaxMicrolineCount)) {
        return;
    }
    // i16(-32768, 32768).
    // x256 anti-aliasing?
    let microlineSubpixels = vec4<i32>(round(clamp(microlineSegment, vec4<f32>(-32768.0), vec4<f32>(32767.0)) * 256.0));
    let microlinePixels = vec4<i32>(floor(vec4<f32>(microlineSubpixels) / 256.0));
    let microlineFractPixels = microlineSubpixels - microlinePixels * 256;

    // Pack.
    bMicrolines.data[outputMicrolineIndex] = uvec4(
        (u32(microlinePixels.x) & 0xffffu) | (u32(microlinePixels.y) << 16u),
        (u32(microlinePixels.z) & 0xffffu) | (u32(microlinePixels.w) << 16u),
        u32(microlineFractPixels.x)        | (u32(microlineFractPixels.y) << 8u) |
        (u32(microlineFractPixels.z) << 16u) | (u32(microlineFractPixels.w) << 24u),
        pathIndex
    );
}

// See Kaspar Fischer, "Piecewise Linear Approximation of Bézier Curves", 2000.
fn curveIsFlat(baseline: vec4<f32>, ctrl: vec4<f32>) -> bool {
    var uv = vec4<f32>(3.0) * ctrl - vec4<f32>(2.0) * baseline - baseline.zwxy;
    uv = uv * uv;
    uv = max(uv, uv.zwxy);
    return uv.x + uv.y <= 16.0 * TOLERANCE * TOLERANCE;
}

// 修复点：WGSL不支持 out 传参，此处重构为标准的指针 ptr 传参
fn subdivideCurve(
    baseline: vec4<f32>,
    ctrl: vec4<f32>,
    t: f32,
    prevBaseline: ptr<function, vec4<f32>>,
    prevCtrl: ptr<function, vec4<f32>>,
    nextBaseline: ptr<function, vec4<f32>>,
    nextCtrl: ptr<function, vec4<f32>>
) {
    let p0 = baseline.xy;
    let p1 = ctrl.xy;
    let p2 = ctrl.zw;
    let p3 = baseline.zw;

    let p0p1 = mix(p0, p1, t);
    let p1p2 = mix(p1, p2, t);
    let p2p3 = mix(p2, p3, t);
    let p0p1p2 = mix(p0p1, p1p2, t);
    let p1p2p3 = mix(p1p2, p2p3, t);
    let p0p1p2p3 = mix(p0p1p2, p1p2p3, t);

    *prevBaseline = vec4<f32>(p0, p0p1p2p3);
    *prevCtrl = vec4<f32>(p0p1, p0p1p2);
    *nextBaseline = vec4<f32>(p0p1p2p3, p3);
    *nextCtrl = vec4<f32>(p1p2p3, p2p3);
}

fn sampleCurve(baseline: vec4<f32>, ctrl: vec4<f32>, t: f32) -> vec2<f32> {
    let p0 = baseline.xy;
    let p1 = ctrl.xy;
    let p2 = ctrl.zw;
    let p3 = baseline.zw;

    let p0p1 = mix(p0, p1, t);
    let p1p2 = mix(p1, p2, t);
    let p2p3 = mix(p2, p3, t);
    let p0p1p2 = mix(p0p1, p1p2, t);
    let p1p2p3 = mix(p1p2, p2p3, t);
    return mix(p0p1p2, p1p2p3, t);
}

fn sampleLine(line_: vec4<f32>, t: f32) -> vec2<f32> {
    return mix(line_.xy, line_.zw, t);
}

fn getPoint(pointIndex: u32) -> vec2<f32> {
    return uUniform0.uTransform * bPoints.data[pointIndex] + uUniform0.uTranslation;
}

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // One path per thread.
    let batchSegmentIndex = global_id.x;
    if (batchSegmentIndex >= u32(uUniform1.uLastBatchSegmentIndex)) {
        return;
    }

    // Find the path index.
    var lowPathIndex = 0u;
    var highPathIndex = u32(uUniform1.uPathCount);
    var iteration = 0;

    while (iteration < 1024 && lowPathIndex + 1u < highPathIndex) {
        let midPathIndex = lowPathIndex + (highPathIndex - lowPathIndex) / 2u;
        // iDiceMetadata.z: first batch segment index
        let midBatchSegmentIndex = bDiceMetadata.data[midPathIndex].z;

        if (batchSegmentIndex < midBatchSegmentIndex) {
            highPathIndex = midPathIndex;
        } else {
            lowPathIndex = midPathIndex;
            if (batchSegmentIndex == midBatchSegmentIndex) {
                break;
            }
        }
        iteration++;
    }

    let batchPathIndex = lowPathIndex;
    // CHY: Fetch the dice metadata of No.batchPathIndex path.
    let diceMetadata = bDiceMetadata.data[batchPathIndex];
    let firstGlobalSegmentIndexInPath = diceMetadata.y;
    let firstBatchSegmentIndexInPath = diceMetadata.z;
    let globalSegmentIndex = batchSegmentIndex - firstBatchSegmentIndexInPath + firstGlobalSegmentIndexInPath;

    let inputIndices = bInputIndices.data[globalSegmentIndex];
    let fromPointIndex = inputIndices.x;
    let flagsPathIndex = inputIndices.y;

    var toPointIndex = fromPointIndex;
    if ((flagsPathIndex & FLAGS_PATH_INDEX_CURVE_IS_CUBIC) != 0u) {
        toPointIndex += 3u;
    } else if ((flagsPathIndex & FLAGS_PATH_INDEX_CURVE_IS_QUADRATIC) != 0u) {
        toPointIndex += 2u;
    } else {
        toPointIndex += 1u;
    }

    // Get start and end point positions by index.
    var baseline = vec4<f32>(getPoint(fromPointIndex), getPoint(toPointIndex));

    // Read control points if applicable, and calculate number of segments.
    //
    // The technique is from Thomas Sederberg, "Computer-Aided Geometric Design" notes, section
    // 10.6 "Error Bounds".
    var ctrl = vec4<f32>(0.0);
    var segmentCountF: f32;
    let isCurve = (flagsPathIndex & (FLAGS_PATH_INDEX_CURVE_IS_CUBIC | FLAGS_PATH_INDEX_CURVE_IS_QUADRATIC)) != 0u;

    if (isCurve) {
        let ctrl0 = getPoint(fromPointIndex + 1u);
        if ((flagsPathIndex & FLAGS_PATH_INDEX_CURVE_IS_QUADRATIC) != 0u) {
            let ctrl0_2 = ctrl0 * 2.0;
            ctrl = (baseline + ctrl0_2.xyxy) * vec4<f32>(1.0 / 3.0);
        } else {
            ctrl = vec4<f32>(ctrl0, getPoint(fromPointIndex + 2u));
        }
        let bound = vec2<f32>(6.0) * max(abs(ctrl.zw - 2.0 * ctrl.xy + baseline.xy),
                                         abs(baseline.zw - 2.0 * ctrl.zw + ctrl.xy));
        segmentCountF = sqrt(length(bound) / (8.0 * TOLERANCE));
    } else {
        segmentCountF = length(baseline.zw - baseline.xy) / MICROLINE_LENGTH;
    }

    // Microline count.
    let segmentCount = max(i32(ceil(segmentCountF)), 1);

    // Update microline_count in the indirect_compute_params.
    // 修复点：将原子累加操作一比一对应转换到 WebGPU 规范中的存储原子操作
    let firstOutputMicrolineIndex = atomicAdd(&bComputeIndirectParams.iComputeIndirectParams[BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX], u32(segmentCount));

    // On-path t of the previous point.
    var prevT = 0.0;
    // CHY: Real coordinates of the previous point.
    var prevPoint = baseline.xy;

    // Do the cut.
    for (var segmentIndex = 0; segmentIndex < segmentCount; segmentIndex++) {
        let nextT = f32(segmentIndex + 1) / f32(segmentCount);
        var nextPoint: vec2<f32>;

        // Sample point on path.
        if (isCurve) {
            nextPoint = sampleCurve(baseline, ctrl, nextT);
        } else {
            nextPoint = sampleLine(baseline, nextT);
        }

        emitMicroline(vec4<f32>(prevPoint, nextPoint), batchPathIndex, firstOutputMicrolineIndex + u32(segmentIndex));
        prevT = nextT;
        prevPoint = nextPoint;
    }
}