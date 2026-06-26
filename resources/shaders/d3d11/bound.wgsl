// pathfinder/resources/shaders/d3d11/bound.wgsl
// Initializes the tile maps.

struct Globals {
    uPathCount: i32,
    uTileCount: i32,
    uPad0: i32,
    uPad1: i32,
};

@group(0) @binding(2) var<uniform> globals: Globals;

struct TilePathInfo {
    // x: tile upper left, 16-bit packed x/y
    // y: tile lower right, 16-bit packed x/y
    // z: first tile index in this path
    // w: color/ctrl/backdrop word
    data: array<vec4<u32>>,
};
@group(1) @binding(0) var<storage, read> bTilePathInfo: TilePathInfo;

struct Tiles {
    // [0]: next tile ID (initialized to -1)
    // [1]: first fill ID (initialized to -1)
    // [2]: backdrop delta upper 8 bits, alpha tile ID lower 24 (initialized to 0, -1 respectively)
    // [3]: color/ctrl/backdrop word
    data: array<u32>,
};
@group(1) @binding(1) var<storage, read_write> bTiles: Tiles;

const TILE_FIELD_NEXT_TILE_ID: u32 = 0u;
const TILE_FIELD_FIRST_FILL_ID: u32 = 1u;
const TILE_FIELD_BACKDROP_ALPHA_TILE_ID: u32 = 2u;
const TILE_FIELD_CONTROL: u32 = 3u;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tileIndex = global_id.x;
    if (tileIndex >= u32(globals.uTileCount)) {
        return;
    }

    var lowPathIndex = 0u;
    var highPathIndex = u32(globals.uPathCount);
    var iteration = 0;

    while (iteration < 1024 && lowPathIndex + 1u < highPathIndex) {
        let midPathIndex = lowPathIndex + (highPathIndex - lowPathIndex) / 2u;
        let midTileIndex = bTilePathInfo.data[midPathIndex].z;

        if (tileIndex < midTileIndex) {
            highPathIndex = midPathIndex;
        } else {
            lowPathIndex = midPathIndex;
            if (tileIndex == midTileIndex) {
                break;
            }
        }
        iteration++;
    }

    let pathIndex = lowPathIndex;
    let pathInfo = bTilePathInfo.data[pathIndex];

    let packedTileRect = vec2<i32>(i32(pathInfo.x), i32(pathInfo.y));

    // 原汁原味保留 16 位有符号整数打包解包的位移操作
    let tileRect = vec4<i32>(
        (packedTileRect.x << 16u) >> 16u,
        packedTileRect.x >> 16u,
        (packedTileRect.y << 16u) >> 16u,
        packedTileRect.y >> 16u
    );

    let tileOffset = tileIndex - pathInfo.z;
    let tileWidth = u32(tileRect.z - tileRect.x);

    // 计算当前 Tile 在大区域中的坐标（注：原 GLSL 计算了 tileCoords 但并没有直接使用，此处一比一保留算法）
    let tileCoords = tileRect.xy + vec2<i32>(i32(tileOffset % tileWidth), i32(tileOffset / tileWidth));

    // 按 4 个 u32 一组的扁平排列格式初始化存储
    bTiles.data[tileIndex * 4u + TILE_FIELD_NEXT_TILE_ID] = ~0u; // 即 0xffffffffu
    bTiles.data[tileIndex * 4u + TILE_FIELD_FIRST_FILL_ID] = ~0u;
    bTiles.data[tileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID] = 0x00ffffffu;
    bTiles.data[tileIndex * 4u + TILE_FIELD_CONTROL] = pathInfo.w;
}