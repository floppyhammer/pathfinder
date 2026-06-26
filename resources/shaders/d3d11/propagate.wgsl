// Sum up backdrops to propagate fills across tiles, and allocate alpha tiles.

// TILE FIELD CONSTANTS
const TILE_FIELD_NEXT_TILE_ID: u32 = 0u;
const TILE_FIELD_FIRST_FILL_ID: u32 = 1u;
const TILE_FIELD_BACKDROP_ALPHA_TILE_ID: u32 = 2u;
const TILE_FIELD_CONTROL: u32 = 3u;

const FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX: u32 = 4u;
const FILL_INDIRECT_DRAW_PARAMS_SIZE: i32 = 8;

const TILE_CTRL_MASK_MASK: u32 = 0x3u;
const TILE_CTRL_MASK_WINDING: u32 = 0x1u;
const TILE_CTRL_MASK_EVEN_ODD: u32 = 0x2u;

const TILE_CTRL_MASK_0_SHIFT: u32 = 0u;

struct BUniform {
    uFramebufferTileSize: vec2<i32>,
    uColumnCount: i32,
    uFirstAlphaTileIndex: i32,
}

struct DrawMetadata {
    // [0]: tile rect
    // [1].x: tile offset
    // [1].y: path ID
    // [1].z: Z write enabled?
    // [1].w: clip path ID, or ~0
    // [2].x: backdrop column offset
    iDrawMetadata: array<vec4<u32>>,
}

struct ClipMetadata {
    // [0]: tile rect
    // [1].x: tile offset
    // [1].y: unused
    // [1].z: unused
    // [1].w: unused
    iClipMetadata: array<vec4<u32>>,
}

struct Backdrops {
    // [0]: backdrop
    // [1]: tile X offset
    // [2]: path ID
    iBackdrops: array<i32>,
}

struct DrawTiles {
    // [0]: next tile ID
    // [1]: first fill ID
    // [2]: backdrop delta upper 8 bits, alpha tile ID lower 24
    // [3]: color/ctrl/backdrop word
    iDrawTiles: array<u32>,
}

struct ClipTiles {
    // [0]: next tile ID
    // [1]: first fill ID
    // [2]: backdrop delta upper 8 bits, alpha tile ID lower 24
    // [3]: color/ctrl/backdrop word
    iClipTiles: array<u32>,
}

struct ZBuffer {
    // [0]: vertexCount (6)
    // [1]: instanceCount (of fills)
    // [2]: vertexStart (0)
    // [3]: baseInstance (0)
    // [4]: alpha tile count
    // [8..]: z-buffer
    iZBuffer: array<atomic<i32>>,
}

struct FirstTileMap {
    iFirstTileMap: array<atomic<i32>>,
}

struct AlphaTiles {
    // [0]: alpha tile index
    // [1]: clip tile index
    iAlphaTiles: array<u32>,
}

// Bindings matched precisely with original order
@group(0) @binding(8) var<uniform> bUniform: BUniform;
@group(0) @binding(0) var<storage, read> bDrawMetadata: DrawMetadata;
@group(0) @binding(1) var<storage, read> bClipMetadata: ClipMetadata;
@group(0) @binding(2) var<storage, read> bBackdrops: Backdrops;
@group(0) @binding(3) var<storage, read_write> bDrawTiles: DrawTiles;
@group(0) @binding(4) var<storage, read> bClipTiles: ClipTiles;
@group(0) @binding(5) var<storage, read_write> bZBuffer: ZBuffer;
@group(0) @binding(6) var<storage, read_write> bFirstTileMap: FirstTileMap;
@group(0) @binding(7) var<storage, read_write> bAlphaTiles: AlphaTiles;

fn calculateTileIndex(bufferOffset: u32, tileRect: vec4<u32>, tileCoord: vec2<u32>) -> u32 {
    return bufferOffset + tileCoord.y * (tileRect.z - tileRect.x) + tileCoord.x;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>) {
    var columnIndex = gl_GlobalInvocationID.x;
    if (i32(columnIndex) >= bUniform.uColumnCount) {
        return;
    }

    var currentBackdrop = bBackdrops.iBackdrops[columnIndex * 3u + 0u];
    var tileX = bBackdrops.iBackdrops[columnIndex * 3u + 1u];
    var drawPathIndex = u32(bBackdrops.iBackdrops[columnIndex * 3u + 2u]);

    var drawTileRect = bDrawMetadata.iDrawMetadata[drawPathIndex * 3u + 0u];
    var drawOffsets = bDrawMetadata.iDrawMetadata[drawPathIndex * 3u + 1u];
    var drawTileSize = drawTileRect.zw - drawTileRect.xy;
    var drawTileBufferOffset = drawOffsets.x;
    var zWrite = drawOffsets.z != 0u;

    var clipPathIndex = i32(drawOffsets.w);
    var clipTileRect = vec4<u32>(0u);
    var clipOffsets = vec4<u32>(0u);
    if (clipPathIndex >= 0) {
        clipTileRect = bClipMetadata.iClipMetadata[u32(clipPathIndex) * 2u + 0u];
        clipOffsets = bClipMetadata.iClipMetadata[u32(clipPathIndex) * 2u + 1u];
    }
    var clipTileBufferOffset = clipOffsets.x;
    var clipBackdropOffset = clipOffsets.y;

    for (var tileY: u32 = 0u; tileY < drawTileSize.y; tileY++) {
        var drawTileCoord = vec2<u32>(u32(tileX), tileY);
        var drawTileIndex = calculateTileIndex(drawTileBufferOffset, drawTileRect, drawTileCoord);

        var drawAlphaTileIndex: i32 = -1;
        var clipAlphaTileIndex: i32 = -1;
        var drawFirstFillIndex = i32(bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_FIRST_FILL_ID]);
        var drawBackdropDelta = i32(bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID]) >> 24;
        var drawTileWord = bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_CONTROL] & 0x00ffffffu;

        var drawTileBackdrop = currentBackdrop;
        var haveDrawAlphaMask = drawFirstFillIndex >= 0;
        var needNewAlphaTile = haveDrawAlphaMask;

        // Handle clip if necessary.
        if (clipPathIndex >= 0) {
            var tileCoord = drawTileCoord + drawTileRect.xy;
            if (all(tileCoord >= clipTileRect.xy) && all(tileCoord < clipTileRect.zw)) {
                var clipTileCoord = tileCoord - clipTileRect.xy;
                var clipTileIndex = calculateTileIndex(clipTileBufferOffset, clipTileRect, clipTileCoord);
                var thisClipAlphaTileIndex = i32(bClipTiles.iClipTiles[clipTileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID] << 8u) >> 8;

                var clipTileWord = bClipTiles.iClipTiles[clipTileIndex * 4u + TILE_FIELD_CONTROL];
                var clipTileBackdrop = i32(clipTileWord) >> 24;

                if (thisClipAlphaTileIndex >= 0) {
                    if (haveDrawAlphaMask) {
                        clipAlphaTileIndex = thisClipAlphaTileIndex;
                        needNewAlphaTile = true;
                    } else {
                        if (drawTileBackdrop != 0) {
                            // This is a solid draw tile, but there's a clip applied.
                            // Replace it with an alpha tile pointing directly to the clip mask.
                            drawAlphaTileIndex = thisClipAlphaTileIndex;
                            clipAlphaTileIndex = -1;
                            needNewAlphaTile = false;
                        } else {
                            // No draw alpha tile index, no clip alpha tile index.
                            drawAlphaTileIndex = -1;
                            clipAlphaTileIndex = -1;
                            needNewAlphaTile = false;
                        }
                    }
                } else {
                    // No clip tile.
                    if (clipTileBackdrop == 0) {
                        // This is a blank clip tile.
                        // Cull the draw tile entirely.
                        drawTileBackdrop = 0;
                        needNewAlphaTile = false;
                    }
                }
            } else {
                // This draw tile is outside the clip path bounding rect.
                // Cull the draw tile.
                drawTileBackdrop = 0;
                needNewAlphaTile = false;
            }
        }

        if (needNewAlphaTile) {
            var drawBatchAlphaTileIndex = u32(atomicAdd(&bZBuffer.iZBuffer[FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX], 1));
            bAlphaTiles.iAlphaTiles[drawBatchAlphaTileIndex * 2u + 0u] = drawTileIndex;
            bAlphaTiles.iAlphaTiles[drawBatchAlphaTileIndex * 2u + 1u] = u32(clipAlphaTileIndex);
            drawAlphaTileIndex = i32(drawBatchAlphaTileIndex) + bUniform.uFirstAlphaTileIndex;
        }

        // Note that drawAlphaTileIndex is signed.
        bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_BACKDROP_ALPHA_TILE_ID] =
            (u32(drawAlphaTileIndex) & 0x00ffffffu) | (u32(drawBackdropDelta) << 24u);
        bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_CONTROL] =
            drawTileWord | (u32(drawTileBackdrop) << 24u);

        // Even-Odd fill rule will make some solid tiles invisible, we shouldn't write them into Z buffer.
        if (drawTileBackdrop != 0) {
            var tileCtrl = i32((drawTileWord >> 16u) & 0xffu);
            var maskCtrl = u32(tileCtrl >> TILE_CTRL_MASK_0_SHIFT) & TILE_CTRL_MASK_MASK;

            if ((maskCtrl & TILE_CTRL_MASK_EVEN_ODD) != 0u && i32(abs(drawTileBackdrop)) % 2 == 0) {
                zWrite = false;
            }
        }

        // Write to Z-buffer if necessary.
        var tileCoord = vec2<i32>(i32(tileX), i32(tileY)) + bUniform.uFramebufferTileSize;
        var tileMapIndex = i32(tileCoord.y) * bUniform.uFramebufferTileSize.x + i32(tileCoord.x);

        if (zWrite && drawTileBackdrop != 0 && drawAlphaTileIndex < 0) {
            atomicMax(&bZBuffer.iZBuffer[u32(tileMapIndex + FILL_INDIRECT_DRAW_PARAMS_SIZE)], i32(drawTileIndex));
        }

        // Stitch into the linked list if necessary.
        if (drawTileBackdrop != 0 || drawAlphaTileIndex >= 0) {
            var nextTileIndex = atomicExchange(&bFirstTileMap.iFirstTileMap[tileMapIndex], i32(drawTileIndex));
            bDrawTiles.iDrawTiles[drawTileIndex * 4u + TILE_FIELD_NEXT_TILE_ID] = u32(nextTileIndex);
        }

        currentBackdrop += drawBackdropDelta;
    }
}