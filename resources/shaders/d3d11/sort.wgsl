// pathfinder/resources/shaders/d3d11/sort.wgsl
// Sorts tiles by Z-buffer depth using an insertion sort algorithm.

struct Uniforms {
    uTileCount: i32,
    uPad0: i32,
    uPad1: i32,
    uPad2: i32,
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

struct Tiles {
    // [0]: next tile ID
    // [1]: first fill ID
    // [2]: backdrop delta upper 8 bits, alpha tile ID lower 24
    // [3]: color/ctrl/backdrop word
    data: array<atomic<u32>>,
};
@group(1) @binding(0) var<storage, read_write> bTiles: Tiles;

struct FirstTileMap {
    data: array<atomic<i32>>,
};
@group(1) @binding(1) var<storage, read_write> bFirstTileMap: FirstTileMap;

struct ZBuffer {
    data: array<i32>,
};
@group(1) @binding(2) var<storage, read> bZBuffer: ZBuffer;

const TILE_FIELD_NEXT_TILE_ID: u32 = 0u;
const TILE_FIELD_FIRST_FILL_ID: u32 = 1u;
const TILE_FIELD_BACKDROP_ALPHA_TILE_ID: u32 = 2u;
const TILE_FIELD_CONTROL: u32 = 3u;

const FILL_INDIRECT_DRAW_PARAMS_SIZE: i32 = 8;

fn getFirst(globalTileIndex: u32) -> i32 {
    return atomicLoad(&bFirstTileMap.data[globalTileIndex]);
}

fn setFirst(globalTileIndex: u32, newFirstTileIndex: i32) {
    atomicStore(&bFirstTileMap.data[globalTileIndex], newFirstTileIndex);
}

fn getNextTile(tileIndex: i32) -> i32 {
    if (tileIndex < 0) {
        return -1;
    }
    return i32(atomicLoad(&bTiles.data[u32(tileIndex) * 4u + TILE_FIELD_NEXT_TILE_ID]));
}

fn setNextTile(tileIndex: i32, newNextTileIndex: i32) {
    if (tileIndex >= 0) {
        atomicStore(&bTiles.data[u32(tileIndex) * 4u + TILE_FIELD_NEXT_TILE_ID], u32(newNextTileIndex));
    }
}

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let globalTileIndex = global_id.x;
    if (globalTileIndex >= u32(uniforms.uTileCount)) {
        return;
    }

    let zValue = bZBuffer.data[FILL_INDIRECT_DRAW_PARAMS_SIZE + i32(globalTileIndex)];

    var unsortedFirstTileIndex = getFirst(globalTileIndex);
    var sortedFirstTileIndex = -1;
    var outerIteration = 0;

    // 遍历未排序的原始链表
    while (unsortedFirstTileIndex >= 0 && outerIteration < 1024) {
        let currentTileIndex = unsortedFirstTileIndex;
        unsortedFirstTileIndex = getNextTile(currentTileIndex);

        if (currentTileIndex >= zValue) {
            var prevTrialTileIndex = -1;
            var trialTileIndex = sortedFirstTileIndex;
            var innerIteration = 0;

            // 寻找新链表中的插入位置
            while (innerIteration < 1024) {
                if (trialTileIndex < 0 || currentTileIndex < trialTileIndex) {
                    if (prevTrialTileIndex < 0) {
                        setNextTile(currentTileIndex, sortedFirstTileIndex);
                        sortedFirstTileIndex = currentTileIndex;
                    } else {
                        setNextTile(currentTileIndex, trialTileIndex);
                        setNextTile(prevTrialTileIndex, currentTileIndex);
                    }
                    break;
                }
                prevTrialTileIndex = trialTileIndex;
                trialTileIndex = getNextTile(trialTileIndex);
                innerIteration++;
            }
        } else {
            // 直接丢弃不满足 Z 值的 Tile 节点，将其闭合
            setNextTile(currentTileIndex, -1);
        }
        outerIteration++;
    }

    // 将排序后重新串联的链表头部写回
    setFirst(globalTileIndex, sortedFirstTileIndex);
}