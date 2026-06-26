// pathfinder/renderer/src/gpu/d3d11/renderer.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A GPU compute-based renderer that uses functionality available in Direct3D 11.

use crate::gpu::perf::TimeCategory;
use crate::gpu::renderer::RendererCore;
use crate::gpu_data::{AlphaTileD3D11, BackdropInfoD3D11, DiceMetadataD3D11, DrawTileBatchD3D11};
use crate::gpu_data::{Fill, FirstTileD3D11, MicrolineD3D11, PathSource, PropagateMetadataD3D11};
use crate::gpu_data::{SegmentIndicesD3D11, SegmentsD3D11, TileBatchDataD3D11, TileD3D11};
use crate::gpu_data::{TileBatchTexture, TilePathInfoD3D11};
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::Vector2F;
use pathfinder_gpu::allocator::{BufferTag, GeneralBufferID, GpuMemoryAllocator};
use pathfinder_gpu::Device;
use pathfinder_resources::ResourceLoader;
use std::mem;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use vec_map::VecMap;
use wgpu::util::DeviceExt;

const FILL_INDIRECT_DRAW_PARAMS_INSTANCE_COUNT_INDEX: usize = 1;
const FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX: usize = 4;
const FILL_INDIRECT_DRAW_PARAMS_SIZE: usize = 8;

const BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX: usize = 3;

const LOAD_ACTION_CLEAR: i32 = 0;
const LOAD_ACTION_LOAD: i32 = 1;

const INITIAL_ALLOCATED_MICROLINE_COUNT: u32 = 1024 * 16;
const INITIAL_ALLOCATED_FILL_COUNT: u32 = 1024 * 16;

pub(crate) const BOUND_WORKGROUP_SIZE: u32 = 64;
pub(crate) const DICE_WORKGROUP_SIZE: u32 = 64;
pub(crate) const BIN_WORKGROUP_SIZE: u32 = 64;
pub(crate) const PROPAGATE_WORKGROUP_SIZE: u32 = 64;
pub(crate) const SORT_WORKGROUP_SIZE: u32 = 64;

pub(crate) struct RendererD3D11 {
    propagate_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,
    tile_pipeline: wgpu::ComputePipeline,
    bin_pipeline: wgpu::ComputePipeline,
    dice_pipeline: wgpu::ComputePipeline,
    bound_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,

    allocated_microline_count: u32,
    allocated_fill_count: u32,
    scene_buffers: SceneBuffers,
    tile_batch_info: VecMap<TileBatchInfoD3D11>,
}

impl RendererD3D11 {
    pub(crate) fn new(core: &RendererCore, resources: &dyn ResourceLoader) -> RendererD3D11 {
        let propagate_pipeline = core
            .device
            .create_compute_pipeline(resources, "d3d11/propagate");
        let fill_pipeline = core.device.create_compute_pipeline(resources, "d3d11/fill");
        let tile_pipeline = core.device.create_compute_pipeline(resources, "d3d11/tile");
        let bin_pipeline = core.device.create_compute_pipeline(resources, "d3d11/bin");
        let dice_pipeline = core.device.create_compute_pipeline(resources, "d3d11/dice");
        let bound_pipeline = core
            .device
            .create_compute_pipeline(resources, "d3d11/bound");
        let sort_pipeline = core.device.create_compute_pipeline(resources, "d3d11/sort");

        RendererD3D11 {
            propagate_pipeline,
            fill_pipeline,
            tile_pipeline,
            bin_pipeline,
            dice_pipeline,
            bound_pipeline,
            allocated_fill_count: INITIAL_ALLOCATED_FILL_COUNT,
            allocated_microline_count: INITIAL_ALLOCATED_MICROLINE_COUNT,
            scene_buffers: SceneBuffers::new(),
            tile_batch_info: VecMap::<TileBatchInfoD3D11>::new(),
            sort_pipeline,
        }
    }

    pub(crate) fn upload_scene(
        &mut self,
        core: &mut RendererCore,
        draw_segments: &SegmentsD3D11,
        clip_segments: &SegmentsD3D11,
    ) {
        self.scene_buffers.upload(
            &mut core.allocator,
            &core.device,
            draw_segments,
            clip_segments,
        );
    }

    pub(crate) fn prepare_and_draw_tiles(
        &mut self,
        core: &mut RendererCore,
        batch: &DrawTileBatchD3D11,
    ) {
        let tile_batch_id = batch.tile_batch_data.batch_id;
        self.prepare_tiles(core, &batch.tile_batch_data);
        let batch_info = self.tile_batch_info[tile_batch_id.0 as usize].clone();
        self.draw_tiles(
            core,
            batch_info.tiles_d3d11_buffer_id,
            batch_info.first_tile_map_buffer_id,
            batch.color_texture,
        );
    }

    pub(crate) fn prepare_tiles(&mut self, core: &mut RendererCore, batch: &TileBatchDataD3D11) {
        core.stats.total_tile_count += batch.tile_count as usize;

        let tiles_d3d11_buffer_id = self.allocate_tiles(core, batch.tile_count);

        let clip_buffer_ids = match batch.clipped_path_info {
            Some(ref clipped_path_info) => {
                let clip_batch_id = clipped_path_info.clip_batch_id;
                let clip_tile_batch_info = &self.tile_batch_info[clip_batch_id.0 as usize];
                let metadata = clip_tile_batch_info.propagate_metadata_buffer_id;
                let tiles = clip_tile_batch_info.tiles_d3d11_buffer_id;
                Some(ClipBufferIDs {
                    metadata: Some(metadata),
                    tiles,
                })
            }
            None => None,
        };

        let z_buffer_id = self.allocate_z_buffer(core);
        let first_tile_map_buffer_id = self.allocate_first_tile_map(core);

        let propagate_metadata_buffer_ids = self.upload_propagate_metadata(
            core,
            &batch.prepare_info.propagate_metadata,
            &batch.prepare_info.backdrops,
        );

        let mut microlines_storage = None;
        for _ in 0..2 {
            microlines_storage = self.dice_segments(
                core,
                &batch.prepare_info.dice_metadata,
                batch.segment_count,
                batch.path_source,
                batch.prepare_info.transform,
            );
            if microlines_storage.is_some() {
                break;
            }
        }
        let microlines_storage =
            microlines_storage.expect("Ran out of space for microlines when dicing!");

        let mut fill_buffer_info = None;
        for _ in 0..2 {
            self.bound(
                core,
                tiles_d3d11_buffer_id,
                batch.tile_count,
                &batch.prepare_info.tile_path_info,
            );

            self.upload_initial_backdrops(
                core,
                propagate_metadata_buffer_ids.backdrops,
                &batch.prepare_info.backdrops,
            );

            fill_buffer_info = self.bin_segments(
                core,
                &microlines_storage,
                &propagate_metadata_buffer_ids,
                tiles_d3d11_buffer_id,
                z_buffer_id,
            );
            if fill_buffer_info.is_some() {
                break;
            }
        }
        let fill_buffer_info = fill_buffer_info.expect("Ran out of space for fills when binning!");

        core.allocator
            .free_general_buffer(microlines_storage.buffer_id);

        let alpha_tiles_buffer_id = self.allocate_alpha_tile_info(core, batch.tile_count);

        let propagate_tiles_info = self.propagate_tiles(
            core,
            batch.prepare_info.backdrops.len() as u32,
            tiles_d3d11_buffer_id,
            z_buffer_id,
            first_tile_map_buffer_id,
            alpha_tiles_buffer_id,
            &propagate_metadata_buffer_ids,
            clip_buffer_ids.as_ref(),
        );

        core.allocator
            .free_general_buffer(propagate_metadata_buffer_ids.backdrops);

        core.reallocate_alpha_tile_pages_if_necessary(true);
        self.draw_fills(
            core,
            &fill_buffer_info,
            tiles_d3d11_buffer_id,
            alpha_tiles_buffer_id,
            &propagate_tiles_info,
        );

        core.allocator
            .free_general_buffer(fill_buffer_info.fill_vertex_buffer_id);
        core.allocator.free_general_buffer(alpha_tiles_buffer_id);

        self.sort_tiles(
            core,
            tiles_d3d11_buffer_id,
            first_tile_map_buffer_id,
            z_buffer_id,
        );

        self.tile_batch_info.insert(
            batch.batch_id.0 as usize,
            TileBatchInfoD3D11 {
                tile_count: batch.tile_count,
                z_buffer_id,
                tiles_d3d11_buffer_id,
                propagate_metadata_buffer_id: propagate_metadata_buffer_ids.propagate_metadata,
                first_tile_map_buffer_id,
            },
        );
    }

    fn allocate_tiles(&mut self, core: &mut RendererCore, tile_count: u32) -> GeneralBufferID {
        core.allocator.allocate_general_buffer::<TileD3D11>(
            &core.device,
            tile_count as u64,
            BufferTag("TilesD3D11"),
        )
    }

    fn allocate_z_buffer(&mut self, core: &mut RendererCore) -> GeneralBufferID {
        let size = core.tile_size().area() as u64 + FILL_INDIRECT_DRAW_PARAMS_SIZE as u64;
        core.allocator
            .allocate_general_buffer::<i32>(&core.device, size, BufferTag("ZBufferD3D11"))
    }

    fn allocate_first_tile_map(&mut self, core: &mut RendererCore) -> GeneralBufferID {
        core.allocator.allocate_general_buffer::<FirstTileD3D11>(
            &core.device,
            core.tile_size().area() as u64,
            BufferTag("FirstTileD3D11"),
        )
    }

    fn allocate_alpha_tile_info(
        &mut self,
        core: &mut RendererCore,
        index_count: u32,
    ) -> GeneralBufferID {
        core.allocator.allocate_general_buffer::<AlphaTileD3D11>(
            &core.device,
            index_count as u64,
            BufferTag("AlphaTileD3D11"),
        )
    }

    fn bound(
        &mut self,
        core: &mut RendererCore,
        tiles_d3d11_buffer_id: GeneralBufferID,
        tile_count: u32,
        tile_path_info: &[TilePathInfoD3D11],
    ) {
        let bound_pipeline = &self.bound_pipeline;

        let path_info_buffer_id = core.allocator.allocate_general_buffer::<TilePathInfoD3D11>(
            &core.device,
            tile_path_info.len() as u64,
            BufferTag("TilePathInfoD3D11"),
        );
        let tile_path_info_buffer = core.allocator.get_general_buffer(path_info_buffer_id);
        core.device
            .upload_to_buffer(tile_path_info_buffer, 0, &tile_path_info);

        let tiles_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BoundGlobals {
            uPathCount: i32,
            uTileCount: i32,
        }

        let globals = BoundGlobals {
            uPathCount: tile_path_info.len() as i32,
            uTileCount: tile_count as i32,
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bound Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bound Bind Group 0"),
                layout: &bound_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bound Bind Group 1"),
                layout: &bound_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tile_path_info_buffer.slice(..).into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tiles_buffer.slice(..).into(),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Bound Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bound Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&bound_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            let workgroup_count = (tile_count + BOUND_WORKGROUP_SIZE - 1) / BOUND_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Other, timer_query);

        core.allocator.free_general_buffer(path_info_buffer_id);
    }

    fn upload_propagate_metadata(
        &mut self,
        core: &mut RendererCore,
        propagate_metadata: &[PropagateMetadataD3D11],
        backdrops: &[BackdropInfoD3D11],
    ) -> PropagateMetadataBufferIDsD3D11 {
        let propagate_metadata_storage_id = core
            .allocator
            .allocate_general_buffer::<PropagateMetadataD3D11>(
                &core.device,
                propagate_metadata.len() as u64,
                BufferTag("PropagateMetadataD3D11"),
            );
        let propagate_metadata_buffer = core
            .allocator
            .get_general_buffer(propagate_metadata_storage_id);
        core.device
            .upload_to_buffer(propagate_metadata_buffer, 0, propagate_metadata);

        let backdrops_storage_id = core.allocator.allocate_general_buffer::<BackdropInfoD3D11>(
            &core.device,
            backdrops.len() as u64,
            BufferTag("BackdropInfoD3D11"),
        );

        PropagateMetadataBufferIDsD3D11 {
            propagate_metadata: propagate_metadata_storage_id,
            backdrops: backdrops_storage_id,
        }
    }

    fn upload_initial_backdrops(
        &self,
        core: &RendererCore,
        backdrops_buffer_id: GeneralBufferID,
        backdrops: &[BackdropInfoD3D11],
    ) {
        let backdrops_buffer = core.allocator.get_general_buffer(backdrops_buffer_id);
        core.device.upload_to_buffer(backdrops_buffer, 0, backdrops);
    }

    fn dice_segments(
        &mut self,
        core: &mut RendererCore,
        dice_metadata: &[DiceMetadataD3D11],
        segment_count: u32,
        path_source: PathSource,
        transform: Transform2F,
    ) -> Option<MicrolinesBufferIDsD3D11> {
        let dice_pipeline = &self.dice_pipeline;

        // First, do all allocations (mutable borrows)
        let buffer_id = core.allocator.allocate_general_buffer::<MicrolineD3D11>(
            &core.device,
            self.allocated_microline_count as u64,
            BufferTag("MicrolinesD3D11"),
        );

        let dice_metadata_buffer_id = core.allocator.allocate_general_buffer::<DiceMetadataD3D11>(
            &core.device,
            dice_metadata.len() as u64,
            BufferTag("DiceMetadataD3D11"),
        );

        // Now, get all buffers (immutable borrows)
        let microlines_buffer = core.allocator.get_general_buffer(buffer_id);

        let scene_source_buffers = match path_source {
            PathSource::Draw => &self.scene_buffers.draw,
            PathSource::Clip => &self.scene_buffers.clip,
        };

        let points_buffer = core
            .allocator
            .get_general_buffer(scene_source_buffers.points_buffer.unwrap());
        let indices_buffer = core
            .allocator
            .get_general_buffer(scene_source_buffers.point_indices_buffer.unwrap());
        let dice_metadata_buffer = core.allocator.get_general_buffer(dice_metadata_buffer_id);
        core.device
            .upload_to_buffer(dice_metadata_buffer, 0, dice_metadata);

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DiceGlobals {
            uTransform: [f32; 4],
            uTranslation: [f32; 2],
            uPathCount: i32,
            uLastBatchSegmentIndex: i32,
            uMaxMicrolineCount: i32,
            _padding: [u8; 4],
        }

        let globals = DiceGlobals {
            uTransform: [
                transform.matrix.0.x(),
                transform.matrix.0.y(),
                transform.matrix.0.z(),
                transform.matrix.0.w(),
            ],
            uTranslation: [transform.vector.0.x(), transform.vector.0.y()],
            uPathCount: dice_metadata.len() as i32,
            uLastBatchSegmentIndex: segment_count as i32,
            uMaxMicrolineCount: self.allocated_microline_count as i32,
            _padding: [0; 4],
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dice Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let indirect_params_buffer = core.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dice Indirect Params"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = core.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dice Readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Dice Bind Group 0"),
                layout: &dice_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Dice Bind Group 1"),
                layout: &dice_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: indirect_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dice_metadata_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: points_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: indices_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: microlines_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dice Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dice Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&dice_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            let workgroup_count = (segment_count + DICE_WORKGROUP_SIZE - 1) / DICE_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&indirect_params_buffer, 0, &readback_buffer, 0, 16);
        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Other, timer_query);

        let microline_count = {
            let mapped = Arc::new(AtomicBool::new(false));
            let mapped_clone = mapped.clone();
            readback_buffer
                .slice(0..16)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped_clone.store(true, Ordering::Release);
                    }
                });
            core.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            while !mapped.load(Ordering::Acquire) {
                core.device
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
            let data = readback_buffer.slice(0..16).get_mapped_range();
            let count = bytemuck::from_bytes::<[u32; 4]>(&data)
                [BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX];
            drop(data);
            readback_buffer.unmap();
            count
        };

        if microline_count > self.allocated_microline_count {
            self.allocated_microline_count = microline_count.next_power_of_two();
            core.allocator.free_general_buffer(buffer_id);
            core.allocator.free_general_buffer(dice_metadata_buffer_id);
            return None;
        }

        core.allocator.free_general_buffer(dice_metadata_buffer_id);
        Some(MicrolinesBufferIDsD3D11 {
            buffer_id,
            count: microline_count,
        })
    }

    fn bin_segments(
        &mut self,
        core: &mut RendererCore,
        microlines_storage: &MicrolinesBufferIDsD3D11,
        propagate_metadata_buffer_ids: &PropagateMetadataBufferIDsD3D11,
        tiles_d3d11_buffer_id: GeneralBufferID,
        z_buffer_id: GeneralBufferID,
    ) -> Option<FillBufferInfoD3D11> {
        let bin_pipeline = &self.bin_pipeline;

        let fill_vertex_buffer_id = core.allocator.allocate_general_buffer::<Fill>(
            &core.device,
            self.allocated_fill_count as u64,
            BufferTag("Fill"),
        );

        let fill_vertex_buffer = core.allocator.get_general_buffer(fill_vertex_buffer_id);
        let microlines_buffer = core
            .allocator
            .get_general_buffer(microlines_storage.buffer_id);
        let tiles_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);
        let propagate_metadata_buffer = core
            .allocator
            .get_general_buffer(propagate_metadata_buffer_ids.propagate_metadata);
        let backdrops_buffer = core
            .allocator
            .get_general_buffer(propagate_metadata_buffer_ids.backdrops);

        let z_buffer = core.allocator.get_general_buffer(z_buffer_id);
        let indirect_draw_params = [6u32, 0, 0, 0, 0, microlines_storage.count, 0, 0];
        core.device
            .upload_to_buffer::<u32>(&z_buffer, 0, &indirect_draw_params);

        let z_readback_buffer = core.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin Z Readback"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BinGlobals {
            uMicrolineCount: i32,
            uMaxFillCount: i32,
        }

        let globals = BinGlobals {
            uMicrolineCount: microlines_storage.count as i32,
            uMaxFillCount: self.allocated_fill_count as i32,
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bin Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bin Bind Group 0"),
                layout: &bin_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bin Bind Group 1"),
                layout: &bin_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: microlines_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: propagate_metadata_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: z_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: fill_vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tiles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: backdrops_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Bin Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bin Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&bin_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            let workgroup_count =
                (microlines_storage.count + BIN_WORKGROUP_SIZE - 1) / BIN_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&z_buffer, 0, &z_readback_buffer, 0, 32);
        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;

        let fill_count = {
            let mapped = Arc::new(AtomicBool::new(false));
            let mapped_clone = mapped.clone();
            z_readback_buffer
                .slice(0..32)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped_clone.store(true, Ordering::Release);
                    }
                });
            core.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            while !mapped.load(Ordering::Acquire) {
                core.device
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
            let data = z_readback_buffer.slice(0..32).get_mapped_range();
            let count = bytemuck::from_bytes::<[u32; 8]>(&data)
                [FILL_INDIRECT_DRAW_PARAMS_INSTANCE_COUNT_INDEX];
            drop(data);
            z_readback_buffer.unmap();
            count
        };

        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Other, timer_query);

        if fill_count > self.allocated_fill_count {
            self.allocated_fill_count = fill_count.next_power_of_two();
            core.allocator.free_general_buffer(fill_vertex_buffer_id);
            return None;
        }

        Some(FillBufferInfoD3D11 {
            fill_vertex_buffer_id,
        })
    }

    fn propagate_tiles(
        &mut self,
        core: &mut RendererCore,
        column_count: u32,
        tiles_d3d11_buffer_id: GeneralBufferID,
        z_buffer_id: GeneralBufferID,
        first_tile_map_buffer_id: GeneralBufferID,
        alpha_tiles_buffer_id: GeneralBufferID,
        propagate_metadata_buffer_ids: &PropagateMetadataBufferIDsD3D11,
        clip_buffer_ids: Option<&ClipBufferIDs>,
    ) -> PropagateTilesInfoD3D11 {
        let propagate_pipeline = &self.propagate_pipeline;

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);
        let propagate_metadata_storage_buffer = core
            .allocator
            .get_general_buffer(propagate_metadata_buffer_ids.propagate_metadata);
        let backdrops_storage_buffer = core
            .allocator
            .get_general_buffer(propagate_metadata_buffer_ids.backdrops);

        let z_buffer = core.allocator.get_general_buffer(z_buffer_id);
        let z_buffer_size = core.tile_size();
        let tile_area = z_buffer_size.area() as usize;
        core.device.upload_to_buffer(
            z_buffer,
            FILL_INDIRECT_DRAW_PARAMS_SIZE * mem::size_of::<i32>(),
            &vec![0i32; tile_area],
        );

        let z_readback_buffer = core.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Propagate Z Readback"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let first_tile_map_storage_buffer =
            core.allocator.get_general_buffer(first_tile_map_buffer_id);
        core.device.upload_to_buffer::<FirstTileD3D11>(
            &first_tile_map_storage_buffer,
            0,
            &vec![FirstTileD3D11::default(); tile_area],
        );

        let alpha_tiles_storage_buffer = core.allocator.get_general_buffer(alpha_tiles_buffer_id);

        let clip_metadata_buffer = match clip_buffer_ids {
            Some(clip_buffer_ids) => {
                let clip_metadata_buffer_id = clip_buffer_ids
                    .metadata
                    .expect("Where's the clip metadata storage?");
                Some(core.allocator.get_general_buffer(clip_metadata_buffer_id))
            }
            None => None,
        };

        let clip_tile_buffer = match clip_buffer_ids {
            Some(clip_buffer_ids) => core.allocator.get_general_buffer(clip_buffer_ids.tiles),
            None => tiles_d3d11_buffer,
        };

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PropagateGlobals {
            uFramebufferTileSize: [i32; 2],
            uColumnCount: i32,
            uFirstAlphaTileIndex: i32,
        }

        let framebuffer_tile_size = core.framebuffer_tile_size().0;
        let globals = PropagateGlobals {
            uFramebufferTileSize: [framebuffer_tile_size.x(), framebuffer_tile_size.y()],
            uColumnCount: column_count as i32,
            uFirstAlphaTileIndex: core.alpha_tile_count as i32,
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Propagate Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Propagate Bind Group 0"),
                layout: &propagate_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Propagate Bind Group 1"),
                layout: &propagate_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: propagate_metadata_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: clip_metadata_buffer
                            .unwrap_or(propagate_metadata_storage_buffer)
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: backdrops_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tiles_d3d11_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: clip_tile_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: z_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: first_tile_map_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: alpha_tiles_storage_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Propagate Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagate Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&propagate_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            let workgroup_count =
                (column_count + PROPAGATE_WORKGROUP_SIZE - 1) / PROPAGATE_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&z_buffer, 0, &z_readback_buffer, 0, 32);
        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;

        let batch_alpha_tile_count = {
            let mapped = Arc::new(AtomicBool::new(false));
            let mapped_clone = mapped.clone();
            z_readback_buffer
                .slice(0..32)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped_clone.store(true, Ordering::Release);
                    }
                });
            core.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            while !mapped.load(Ordering::Acquire) {
                core.device
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
            let data = z_readback_buffer.slice(0..32).get_mapped_range();
            let count = bytemuck::from_bytes::<[u32; 8]>(&data)
                [FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX];
            drop(data);
            z_readback_buffer.unmap();
            count
        };

        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Other, timer_query);

        let alpha_tile_start = core.alpha_tile_count;
        core.alpha_tile_count += batch_alpha_tile_count;
        core.stats.alpha_tile_count += batch_alpha_tile_count as usize;
        let alpha_tile_end = core.alpha_tile_count;

        PropagateTilesInfoD3D11 {
            alpha_tile_range: alpha_tile_start..alpha_tile_end,
        }
    }

    fn draw_fills(
        &mut self,
        core: &mut RendererCore,
        fill_buffer_info: &FillBufferInfoD3D11,
        tiles_d3d11_buffer_id: GeneralBufferID,
        alpha_tiles_buffer_id: GeneralBufferID,
        propagate_tiles_info: &PropagateTilesInfoD3D11,
    ) {
        let fill_pipeline = &self.fill_pipeline;

        let fill_vertex_buffer = core
            .allocator
            .get_general_buffer(fill_buffer_info.fill_vertex_buffer_id);
        let tiles_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);
        let alpha_tiles_buffer = core.allocator.get_general_buffer(alpha_tiles_buffer_id);

        let area_lut_texture = core.allocator.get_texture(core.area_lut_texture_id);

        let mask_storage = core
            .mask_storage
            .as_ref()
            .expect("Where's the mask storage?");
        let mask_texture = core.allocator.get_texture(mask_storage.texture_id);

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FillAlphaTileRange {
            start: i32,
            end: i32,
        }

        let alpha_tile_range = FillAlphaTileRange {
            start: propagate_tiles_info.alpha_tile_range.start as i32,
            end: propagate_tiles_info.alpha_tile_range.end as i32,
        };

        let alpha_tile_range_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Alpha Tile Range"),
                    contents: bytemuck::cast_slice(&[alpha_tile_range]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let sampler = core
            .device
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fill Bind Group 0"),
                layout: &fill_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&area_lut_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: alpha_tile_range_buffer.as_entire_binding(),
                    },
                ],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fill Bind Group 1"),
                layout: &fill_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: fill_vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tiles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: alpha_tiles_buffer.as_entire_binding(),
                    },
                ],
            });

        let alpha_tile_count =
            propagate_tiles_info.alpha_tile_range.end - propagate_tiles_info.alpha_tile_range.start;
        let workgroup_count_x = (alpha_tile_count + 255) / 256;
        let workgroup_count_y = 1;

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fill Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fill Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&fill_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Fill, timer_query);

        core.mask_storage_flags
            .insert(crate::gpu::renderer::MaskStorageFlags::MASK_TEXTURE_IS_DIRTY);
    }

    fn sort_tiles(
        &mut self,
        core: &mut RendererCore,
        tiles_d3d11_buffer_id: GeneralBufferID,
        first_tile_map_buffer_id: GeneralBufferID,
        z_buffer_id: GeneralBufferID,
    ) {
        let sort_pipeline = &self.sort_pipeline;

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);
        let first_tile_map_buffer = core.allocator.get_general_buffer(first_tile_map_buffer_id);
        let z_buffer = core.allocator.get_general_buffer(z_buffer_id);

        let tile_count = core.framebuffer_tile_size().area();

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SortGlobals {
            uTileCount: i32,
        }

        let globals = SortGlobals {
            uTileCount: tile_count as i32,
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Sort Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sort Bind Group 0"),
                layout: &sort_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sort Bind Group 1"),
                layout: &sort_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tiles_d3d11_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: first_tile_map_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: z_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sort Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sort Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(sort_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);

            let workgroup_count =
                (tile_count as u32 + SORT_WORKGROUP_SIZE - 1) / SORT_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Other, timer_query);
    }

    pub(crate) fn draw_tiles(
        &mut self,
        core: &mut RendererCore,
        tiles_d3d11_buffer_id: GeneralBufferID,
        first_tile_map_buffer_id: GeneralBufferID,
        color_texture_0: Option<TileBatchTexture>,
    ) {
        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        let tile_pipeline = &self.tile_pipeline;
        let device = &core.device.device;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileGlobals {
            uClearColor: [f32; 4],
            uLoadAction: i32,
            _padding1: [i32; 3],
            uTileSize: [f32; 2],
            uTextureMetadataSize: [i32; 2],
            uColorTextureSize0: [f32; 2],
            uMaskTextureSize0: [f32; 2],
            uFramebufferSize: [f32; 2],
            uFramebufferTileSize: [i32; 2],
        }

        let clear_color = core.clear_color_for_draw_operation();
        let load_action = match clear_color {
            None => LOAD_ACTION_LOAD,
            Some(_) => LOAD_ACTION_CLEAR,
        };

        let draw_viewport = core.draw_viewport();
        let mask_storage = core.mask_storage.as_ref().unwrap();
        let mask_texture = core.allocator.get_texture(mask_storage.texture_id);

        let framebuffer_tile_size = core.framebuffer_tile_size().0;

        let globals = TileGlobals {
            uClearColor: match clear_color {
                Some(c) => [c.r(), c.g(), c.b(), c.a()],
                None => [0.0, 0.0, 0.0, 0.0],
            },
            uLoadAction: load_action,
            _padding1: [0; 3],
            uTileSize: [
                crate::tiles::TILE_WIDTH as f32,
                crate::tiles::TILE_HEIGHT as f32,
            ],
            uTextureMetadataSize: [1024, 1024],
            uColorTextureSize0: [1024.0, 1024.0],
            uMaskTextureSize0: [mask_texture.size.x() as f32, mask_texture.size.y() as f32],
            uFramebufferSize: [
                draw_viewport.size().x() as f32,
                draw_viewport.size().y() as f32,
            ],
            uFramebufferTileSize: [framebuffer_tile_size.x(), framebuffer_tile_size.y()],
        };

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Globals"),
            contents: bytemuck::cast_slice(&[globals]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let dest_texture = core
            .allocator
            .get_texture(core.intermediate_dest_texture_id);

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Bind Group 0"),
            layout: &tile_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Bind Group 1"),
            layout: &tile_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&dest_texture.view),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let metadata_texture = core.allocator.get_texture(core.texture_metadata_texture_id);
        let gamma_lut_texture = core.allocator.get_texture(core.gamma_lut_texture_id);

        let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Bind Group 2"),
            layout: &tile_pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&metadata_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&gamma_lut_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id);
        let first_tile_map_buffer = core.allocator.get_general_buffer(first_tile_map_buffer_id);

        let bind_group_3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Bind Group 3"),
            layout: &tile_pipeline.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tiles_d3d11_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: first_tile_map_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_dimensions_x = framebuffer_tile_size.x() as u32;
        let compute_dimensions_y = framebuffer_tile_size.y() as u32;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&tile_pipeline);
            compute_pass.set_bind_group(0, &bind_group_0, &[]);
            compute_pass.set_bind_group(1, &bind_group_1, &[]);
            compute_pass.set_bind_group(2, &bind_group_2, &[]);
            compute_pass.set_bind_group(3, &bind_group_3, &[]);

            compute_pass.dispatch_workgroups(compute_dimensions_x, compute_dimensions_y, 1);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Composite, timer_query);

        core.preserve_draw_framebuffer();
    }

    pub(crate) fn end_frame(&mut self, core: &mut RendererCore) {
        self.free_tile_batch_buffers(core);
    }

    fn free_tile_batch_buffers(&mut self, core: &mut RendererCore) {
        for (_, tile_batch_info) in self.tile_batch_info.drain() {
            core.allocator
                .free_general_buffer(tile_batch_info.z_buffer_id);
            core.allocator
                .free_general_buffer(tile_batch_info.tiles_d3d11_buffer_id);
            core.allocator
                .free_general_buffer(tile_batch_info.propagate_metadata_buffer_id);
            core.allocator
                .free_general_buffer(tile_batch_info.first_tile_map_buffer_id);
        }
    }
}

#[derive(Clone)]
struct TileBatchInfoD3D11 {
    tile_count: u32,
    z_buffer_id: GeneralBufferID,
    tiles_d3d11_buffer_id: GeneralBufferID,
    propagate_metadata_buffer_id: GeneralBufferID,
    first_tile_map_buffer_id: GeneralBufferID,
}

#[derive(Clone)]
struct FillBufferInfoD3D11 {
    fill_vertex_buffer_id: GeneralBufferID,
}

#[derive(Debug)]
struct PropagateMetadataBufferIDsD3D11 {
    propagate_metadata: GeneralBufferID,
    backdrops: GeneralBufferID,
}

struct MicrolinesBufferIDsD3D11 {
    buffer_id: GeneralBufferID,
    count: u32,
}

#[derive(Clone, Debug)]
struct ClipBufferIDs {
    metadata: Option<GeneralBufferID>,
    tiles: GeneralBufferID,
}

struct SceneBuffers {
    draw: SceneSourceBuffers,
    clip: SceneSourceBuffers,
}

struct SceneSourceBuffers {
    points_buffer: Option<GeneralBufferID>,
    points_capacity: u32,
    point_indices_buffer: Option<GeneralBufferID>,
    point_indices_count: u32,
    point_indices_capacity: u32,
}

#[derive(Clone)]
struct PropagateTilesInfoD3D11 {
    alpha_tile_range: Range<u32>,
}

impl SceneBuffers {
    fn new() -> SceneBuffers {
        SceneBuffers {
            draw: SceneSourceBuffers::new(),
            clip: SceneSourceBuffers::new(),
        }
    }

    fn upload(
        &mut self,
        allocator: &mut GpuMemoryAllocator,
        device: &Device,
        draw_segments: &SegmentsD3D11,
        clip_segments: &SegmentsD3D11,
    ) {
        self.draw.upload(allocator, device, draw_segments);
        self.clip.upload(allocator, device, clip_segments);
    }
}

impl SceneSourceBuffers {
    fn new() -> SceneSourceBuffers {
        SceneSourceBuffers {
            points_buffer: None,
            points_capacity: 0,
            point_indices_buffer: None,
            point_indices_count: 0,
            point_indices_capacity: 0,
        }
    }

    fn upload(
        &mut self,
        allocator: &mut GpuMemoryAllocator,
        device: &Device,
        segments: &SegmentsD3D11,
    ) {
        let needed_points_capacity = (segments.points.len() as u32).next_power_of_two();
        let needed_point_indices_capacity = (segments.indices.len() as u32).next_power_of_two();
        if self.points_capacity < needed_points_capacity {
            self.points_buffer = Some(allocator.allocate_general_buffer::<Vector2F>(
                device,
                needed_points_capacity as u64,
                BufferTag("PointsD3D11"),
            ));
            self.points_capacity = needed_points_capacity;
        }
        if self.point_indices_capacity < needed_point_indices_capacity {
            self.point_indices_buffer =
                Some(allocator.allocate_general_buffer::<SegmentIndicesD3D11>(
                    device,
                    needed_point_indices_capacity as u64,
                    BufferTag("PointIndicesD3D11"),
                ));
            self.point_indices_capacity = needed_point_indices_capacity;
        }
        device.upload_to_buffer(
            allocator.get_general_buffer(self.points_buffer.unwrap()),
            0,
            &segments.points,
        );
        device.upload_to_buffer(
            allocator.get_general_buffer(self.point_indices_buffer.unwrap()),
            0,
            &segments.indices,
        );
        self.point_indices_count = segments.indices.len() as u32;
    }
}
