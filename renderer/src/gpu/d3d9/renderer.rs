// pathfinder/renderer/src/gpu/d3d9/renderer.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A hybrid CPU-GPU renderer that only relies on functionality available in Direct3D 9.

use crate::gpu::renderer::RendererCore;
use crate::gpu::renderer::{MaskStorageFlags, MASK_TEXTURE_HEIGHT, MASK_TEXTURE_WIDTH};
use crate::gpu_data::{Clip, DrawTileBatchD3D9, Fill, TileBatchTexture, TileObjectPrimitive};
use crate::tile_map::DenseTileMap;
use crate::tiles::{TILE_HEIGHT, TILE_WIDTH};
use byte_slice_cast::AsByteSlice;
use pathfinder_color::ColorF;
use pathfinder_content::effects::BlendMode;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::transform3d::Transform4F;
use pathfinder_geometry::vector::{vec2i, Vector2I, Vector4F};
use pathfinder_gpu::allocator::{BufferTag, GeneralBufferID, IndexBufferID, TextureID, TextureTag};
use pathfinder_resources::ResourceLoader;
use wgpu::util::DeviceExt;
use crate::gpu::perf::TimeCategory;

const MAX_FILLS_PER_BATCH: usize = 0x10000;

pub(crate) struct RendererD3D9 {
    // Basic data
    fill_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    // tile_clip_copy_pipeline: wgpu::RenderPipeline,
    // tile_clip_combine_pipeline: wgpu::RenderPipeline,
    // tile_copy_pipeline: wgpu::RenderPipeline,
    quads_vertex_indices_buffer_id: Option<IndexBufferID>,
    quads_vertex_indices_length: usize,

    // Fills.
    buffered_fills: Vec<Fill>,
    pending_fills: Vec<Fill>,

    // Temporary texture
    dest_blend_texture_id: TextureID,
}

impl RendererD3D9 {
    pub(crate) fn new(core: &mut RendererCore, resources: &dyn ResourceLoader) -> RendererD3D9 {
        let fill_pipeline = core
            .device
            .create_render_pipeline(resources, "d3d9/fill", None);
        let tile_pipeline = core
            .device
            .create_render_pipeline(resources, "d3d9/tile", None);
        // let tile_clip_combine_pipeline = core
        //     .device
        //     .create_render_pipeline(resources, "d3d9/tile_clip_combine", None);
        // let tile_clip_copy_pipeline = core
        //     .device
        //     .create_render_pipeline(resources, "d3d9/tile_clip_copy", None);
        // let tile_copy_pipeline = core
        //     .device
        //     .create_render_pipeline(resources, "d3d9/tile_copy", None);

        let window_size = core.options.dest.window_size(&core.device);
        let dest_blend_texture_id = core.allocator.allocate_texture(
            &core.device,
            window_size,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            TextureTag("DestBlendD3D9"),
        );

        RendererD3D9 {
            fill_pipeline,
            tile_pipeline,
            // tile_clip_copy_pipeline,
            // tile_clip_combine_pipeline,
            // tile_copy_pipeline,
            quads_vertex_indices_buffer_id: None,
            quads_vertex_indices_length: 0,

            buffered_fills: vec![],
            pending_fills: vec![],

            dest_blend_texture_id,
        }
    }

    pub(crate) fn upload_and_draw_tiles(
        &mut self,
        core: &mut RendererCore,
        batch: &DrawTileBatchD3D9,
    ) {
        // if !batch.clips.is_empty() {
        //     let clip_buffer_info = self.upload_clip_tiles(core, &batch.clips);
        //     self.clip_tiles(core, &clip_buffer_info);
        //     core.allocator
        //         .free_general_buffer(clip_buffer_info.clip_buffer_id);
        // }

        let tile_buffer = self.upload_tiles(core, &batch.tiles);
        let z_buffer_texture_id = self.upload_z_buffer(core, &batch.z_buffer_data);

        self.draw_tiles(
            core,
            batch.tiles.len() as u32,
            tile_buffer.tile_vertex_buffer_id,
            batch.color_texture,
            batch.blend_mode,
            z_buffer_texture_id,
        );

        core.allocator.free_texture(z_buffer_texture_id);
        core.allocator
            .free_general_buffer(tile_buffer.tile_vertex_buffer_id);
    }

    fn upload_tiles(
        &mut self,
        core: &mut RendererCore,
        tiles: &[TileObjectPrimitive],
    ) -> TileBufferD3D9 {
        let tile_vertex_buffer_id = core
            .allocator
            .allocate_general_buffer::<TileObjectPrimitive>(
                &core.device,
                tiles.len() as u64,
                BufferTag("TileD3D9"),
            );
        let tile_vertex_buffer = &core.allocator.get_general_buffer(tile_vertex_buffer_id);
        core.device.upload_to_buffer(tile_vertex_buffer, 0, tiles);
        self.ensure_index_buffer(core, tiles.len());

        TileBufferD3D9 {
            tile_vertex_buffer_id,
        }
    }

    fn ensure_index_buffer(&mut self, core: &mut RendererCore, mut length: usize) {
        length = length.next_power_of_two();
        if self.quads_vertex_indices_length >= length {
            return;
        }

        let mut indices: Vec<u32> = Vec::with_capacity(length * 6);
        for index in 0..(length as u32) {
            indices.extend_from_slice(&[
                index * 4 + 0,
                index * 4 + 1,
                index * 4 + 2,
                index * 4 + 1,
                index * 4 + 3,
                index * 4 + 2,
            ]);
        }

        if let Some(quads_vertex_indices_buffer_id) = self.quads_vertex_indices_buffer_id.take() {
            core.allocator
                .free_index_buffer(quads_vertex_indices_buffer_id);
        }
        let quads_vertex_indices_buffer_id = core.allocator.allocate_index_buffer::<u32>(
            &core.device,
            indices.len() as u64,
            BufferTag("QuadsVertexIndicesD3D9"),
        );
        let quads_vertex_indices_buffer = core
            .allocator
            .get_index_buffer(quads_vertex_indices_buffer_id);
        core.device
            .upload_to_buffer(quads_vertex_indices_buffer, 0, &indices);
        self.quads_vertex_indices_buffer_id = Some(quads_vertex_indices_buffer_id);
        self.quads_vertex_indices_length = length;
    }

    pub(crate) fn add_fills(&mut self, core: &mut RendererCore, fill_batch: &[Fill]) {
        if fill_batch.is_empty() {
            return;
        }

        core.stats.fill_count += fill_batch.len();

        let preserve_alpha_mask_contents = core.alpha_tile_count > 0;

        self.pending_fills.reserve(fill_batch.len());
        for fill in fill_batch {
            core.alpha_tile_count = core.alpha_tile_count.max(fill.link + 1);
            self.pending_fills.push(*fill);
        }

        core.stats.alpha_tile_count = core.alpha_tile_count as usize;

        core.reallocate_alpha_tile_pages_if_necessary(preserve_alpha_mask_contents);

        if self.buffered_fills.len() + self.pending_fills.len() > MAX_FILLS_PER_BATCH {
            self.draw_buffered_fills(core);
        }

        self.buffered_fills.extend(self.pending_fills.drain(..));
    }

    pub(crate) fn draw_buffered_fills(&mut self, core: &mut RendererCore) {
        if self.buffered_fills.is_empty() {
            return;
        }

        let fill_storage_info = self.upload_buffered_fills(core);
        self.draw_fills(
            core,
            fill_storage_info.fill_buffer_id,
            fill_storage_info.fill_count,
        );
        core.allocator
            .free_general_buffer(fill_storage_info.fill_buffer_id);
    }

    fn upload_buffered_fills(&mut self, core: &mut RendererCore) -> FillBufferInfoD3D9 {
        let buffered_fills = &mut self.buffered_fills;
        debug_assert!(!buffered_fills.is_empty());

        let fill_buffer_id = core.allocator.allocate_general_buffer::<Fill>(
            &core.device,
            MAX_FILLS_PER_BATCH as u64,
            BufferTag("Fill"),
        );
        let fill_vertex_buffer = core.allocator.get_general_buffer(fill_buffer_id);
        debug_assert!(buffered_fills.len() <= u32::MAX as usize);
        core.device
            .upload_to_buffer(fill_vertex_buffer, 0, &buffered_fills);

        let fill_count = buffered_fills.len() as u32;
        buffered_fills.clear();

        FillBufferInfoD3D9 {
            fill_buffer_id,
            fill_count,
        }
    }

    fn draw_fills(
        &mut self,
        core: &mut RendererCore,
        fill_buffer_id: GeneralBufferID,
        fill_count: u32,
    ) {
        let fill_vertex_buffer = core.allocator.get_general_buffer(fill_buffer_id);
        let quad_vertex_positions_buffer = core
            .allocator
            .get_general_buffer(core.quad_vertex_positions_buffer_id);
        let quad_vertex_indices_buffer = core
            .allocator
            .get_index_buffer(core.quad_vertex_indices_buffer_id);

        let area_lut_texture = core.allocator.get_texture(core.area_lut_texture_id);

        let mask_viewport = self.mask_viewport(core);
        let mask_storage = core
            .mask_storage
            .as_ref()
            .expect("Where's the mask storage?");
        let mask_texture_id = mask_storage.texture_id;
        let mask_texture = core.allocator.get_texture(mask_texture_id);

        let mut clear_color = None;
        if !core
            .mask_storage_flags
            .contains(MaskStorageFlags::MASK_TEXTURE_IS_DIRTY)
        {
            clear_color = Some(ColorF::default());
        };

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        // Prepare uniforms
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FillGlobals {
            tile_size: [f32; 2],
            mask_size: [f32; 2],
        }

        let globals = FillGlobals {
            tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
            mask_size: [
                mask_viewport.size().x() as f32,
                mask_viewport.size().y() as f32,
            ],
        };

        let globals_buffer =
            core.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Globals"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let area_lut_view = &area_lut_texture.view;
        let sampler = core
            .device
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.fill_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: globals_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(area_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        let mut encoder =
            core.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fill Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fill Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mask_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match clear_color {
                            Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
                                r: c.r() as f64,
                                g: c.g() as f64,
                                b: c.b() as f64,
                                a: c.a() as f64,
                            }),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.fill_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, quad_vertex_positions_buffer.slice(..));
            render_pass.set_vertex_buffer(1, fill_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                quad_vertex_indices_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.set_viewport(
                0.0,
                0.0,
                mask_viewport.size().x() as f32,
                mask_viewport.size().y() as f32,
                0.0,
                1.0,
            );
            render_pass.draw_indexed(0..6, 0, 0..fill_count);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Fill, timer_query);
        core.mask_storage_flags
            .insert(MaskStorageFlags::MASK_TEXTURE_IS_DIRTY);
    }

    // fn clip_tiles(&mut self, core: &mut RendererCore, clip_buffer_info: &ClipBufferInfo) {
    //     let device = &core.device.device;
    //     let mask_viewport = self.mask_viewport(core);
    //
    //     #[repr(C)]
    //     #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    //     struct ClipGlobals {
    //         framebuffer_size: [f32; 2],
    //     }
    //
    //     let globals = ClipGlobals {
    //         framebuffer_size: [
    //             mask_viewport.size().x() as f32,
    //             mask_viewport.size().y() as f32,
    //         ],
    //     };
    //
    //     let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("Clip Globals"),
    //         contents: bytemuck::cast_slice(&[globals]),
    //         usage: wgpu::BufferUsages::UNIFORM,
    //     });
    //
    //     let mask_storage = core
    //         .mask_storage
    //         .as_ref()
    //         .expect("Where's the mask storage?");
    //     let mask_framebuffer = core.allocator.get_texture(mask_storage.texture_id);
    //     let mask_render_view = mask_framebuffer.create_default_view();
    //     let mask_sample_view = mask_framebuffer.create_default_view();
    //     let sampler = core
    //         .device
    //         .device
    //         .create_sampler(&wgpu::SamplerDescriptor::default());
    //     let clip_buffer = core
    //         .allocator
    //         .get_general_buffer(clip_buffer_info.clip_buffer_id);
    //     let quad_vertex_positions_buffer = core
    //         .allocator
    //         .get_general_buffer(core.quad_vertex_positions_buffer_id);
    //     let quad_vertex_indices_buffer = core
    //         .allocator
    //         .get_index_buffer(core.quad_vertex_indices_buffer_id);
    //
    //     // 1. Copy tiles
    //     {
    //         let copy_pipeline = &self.tile_clip_copy_pipeline;
    //         let bind_group_0 = core
    //             .device
    //             .device
    //             .create_bind_group(&wgpu::BindGroupDescriptor {
    //                 label: None,
    //                 layout: &copy_pipeline.get_bind_group_layout(0),
    //                 entries: &[wgpu::BindGroupEntry {
    //                     binding: 0,
    //                     resource: globals_buffer.as_entire_binding(),
    //                 }],
    //             });
    //         let bind_group_1 = core
    //             .device
    //             .device
    //             .create_bind_group(&wgpu::BindGroupDescriptor {
    //                 label: None,
    //                 layout: &copy_pipeline.get_bind_group_layout(1),
    //                 entries: &[
    //                     wgpu::BindGroupEntry {
    //                         binding: 0,
    //                         resource: wgpu::BindingResource::TextureView(&mask_sample_view),
    //                     },
    //                     wgpu::BindGroupEntry {
    //                         binding: 1,
    //                         resource: wgpu::BindingResource::Sampler(&sampler),
    //                     },
    //                 ],
    //             });
    //
    //         let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //             label: Some("Clip Copy Encoder"),
    //         });
    //         {
    //             let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    //                 label: Some("Clip Copy Pass"),
    //                 color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    //                     view: &mask_render_view,
    //                     resolve_target: None,
    //                     ops: wgpu::Operations {
    //                         load: wgpu::LoadOp::Load,
    //                         store: wgpu::StoreOp::Store,
    //                     },
    //                     depth_slice: None,
    //                 })],
    //                 depth_stencil_attachment: None,
    //                 timestamp_writes: None,
    //                 occlusion_query_set: None,
    //                 multiview_mask: None,
    //             });
    //
    //             render_pass.set_pipeline(&copy_pipeline);
    //             render_pass.set_bind_group(0, &bind_group_0, &[]);
    //             render_pass.set_bind_group(1, &bind_group_1, &[]);
    //             render_pass.set_vertex_buffer(0, quad_vertex_positions_buffer.slice(..));
    //             render_pass.set_vertex_buffer(1, clip_buffer.slice(..));
    //             render_pass.set_index_buffer(
    //                 quad_vertex_indices_buffer.slice(..),
    //                 wgpu::IndexFormat::Uint32,
    //             );
    //             render_pass.draw_indexed(0..6, 0, 0..clip_buffer_info.clip_count);
    //         }
    //         core.device.queue.submit(Some(encoder.finish()));
    //     }
    //
    //     // 2. Combine tiles
    //     {
    //         let combine_pipeline = &self.tile_clip_combine_pipeline;
    //         let bind_group_0 = core
    //             .device
    //             .device
    //             .create_bind_group(&wgpu::BindGroupDescriptor {
    //                 label: None,
    //                 layout: &combine_pipeline.get_bind_group_layout(0),
    //                 entries: &[wgpu::BindGroupEntry {
    //                     binding: 0,
    //                     resource: globals_buffer.as_entire_binding(),
    //                 }],
    //             });
    //         let bind_group_1 = core
    //             .device
    //             .device
    //             .create_bind_group(&wgpu::BindGroupDescriptor {
    //                 label: None,
    //                 layout: &combine_pipeline.get_bind_group_layout(1),
    //                 entries: &[
    //                     wgpu::BindGroupEntry {
    //                         binding: 0,
    //                         resource: wgpu::BindingResource::TextureView(&mask_sample_view),
    //                     },
    //                     wgpu::BindGroupEntry {
    //                         binding: 1,
    //                         resource: wgpu::BindingResource::Sampler(&sampler),
    //                     },
    //                 ],
    //             });
    //
    //         let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //             label: Some("Clip Combine Encoder"),
    //         });
    //         {
    //             let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    //                 label: Some("Clip Combine Pass"),
    //                 color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    //                     view: &mask_render_view,
    //                     resolve_target: None,
    //                     ops: wgpu::Operations {
    //                         load: wgpu::LoadOp::Load,
    //                         store: wgpu::StoreOp::Store,
    //                     },
    //                     depth_slice: None,
    //                 })],
    //                 depth_stencil_attachment: None,
    //                 timestamp_writes: None,
    //                 occlusion_query_set: None,
    //                 multiview_mask: None,
    //             });
    //
    //             render_pass.set_pipeline(&combine_pipeline);
    //             render_pass.set_bind_group(0, &bind_group_0, &[]);
    //             render_pass.set_bind_group(1, &bind_group_1, &[]);
    //             render_pass.set_vertex_buffer(0, quad_vertex_positions_buffer.slice(..));
    //             render_pass.set_vertex_buffer(1, clip_buffer.slice(..));
    //             render_pass.set_index_buffer(
    //                 quad_vertex_indices_buffer.slice(..),
    //                 wgpu::IndexFormat::Uint32,
    //             );
    //             render_pass.draw_indexed(0..6, 0, 0..clip_buffer_info.clip_count);
    //         }
    //         core.device.queue.submit(Some(encoder.finish()));
    //     }
    //
    //     core.stats.drawcall_count += 2;
    // }

    fn upload_z_buffer(
        &mut self,
        core: &mut RendererCore,
        z_buffer_map: &DenseTileMap<i32>,
    ) -> TextureID {
        let z_buffer_texture_id = core.allocator.allocate_texture(
            &core.device,
            z_buffer_map.rect.size(),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            TextureTag("ZBufferD3D9"),
        );
        let z_buffer_texture = core.allocator.get_texture(z_buffer_texture_id);
        debug_assert_eq!(z_buffer_map.rect.origin(), Vector2I::default());
        let z_data: &[u8] = z_buffer_map.data.as_byte_slice();
        core.device.upload_to_texture(
            z_buffer_texture,
            z_buffer_map.rect,
            pathfinder_gpu::TextureDataRef::U8(&z_data),
        );
        z_buffer_texture_id
    }

    fn upload_clip_tiles(&mut self, core: &mut RendererCore, clips: &[Clip]) -> ClipBufferInfo {
        let clip_buffer_id = core.allocator.allocate_general_buffer::<Clip>(
            &core.device,
            clips.len() as u64,
            BufferTag("ClipD3D9"),
        );
        let clip_buffer = core.allocator.get_general_buffer(clip_buffer_id);
        core.device.upload_to_buffer(clip_buffer, 0, clips);
        ClipBufferInfo {
            clip_buffer_id,
            clip_count: clips.len() as u32,
        }
    }

    fn draw_tiles(
        &mut self,
        core: &mut RendererCore,
        tile_count: u32,
        tile_vertex_buffer_id: GeneralBufferID,
        _color_texture_0: Option<TileBatchTexture>,
        _blend_mode: BlendMode,
        z_buffer_texture_id: TextureID,
    ) {
        if tile_count == 0 {
            return;
        }

        let mut timer_query = core
            .timer_query_cache
            .start_timing_draw_call(&core.device, &core.options);

        let tile_pipeline = &self.tile_pipeline;
        let device = &core.device.device;

        // 1. Prepare Tile Globals
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileGlobals {
            tile_size: [f32; 2],
            texture_metadata_size: [i32; 2],
            z_buffer_size: [i32; 2],
            mask_texture_size0: [f32; 2],
            color_texture_size0: [f32; 2],
            framebuffer_size: [f32; 2],
            transform: [f32; 16],
        }

        let transform = self.tile_transform(core);
        let draw_viewport = core.draw_viewport();
        let mask_viewport = self.mask_viewport(core);

        let metadata_texture = core.allocator.get_texture(core.texture_metadata_texture_id);
        let z_buffer_texture = core.allocator.get_texture(z_buffer_texture_id);

        let globals = TileGlobals {
            transform: [
                transform.c0.x(),
                transform.c0.y(),
                transform.c0.z(),
                transform.c0.w(),
                transform.c1.x(),
                transform.c1.y(),
                transform.c1.z(),
                transform.c1.w(),
                transform.c2.x(),
                transform.c2.y(),
                transform.c2.z(),
                transform.c2.w(),
                transform.c3.x(),
                transform.c3.y(),
                transform.c3.z(),
                transform.c3.w(),
            ],
            tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as i32 as f32],
            framebuffer_size: [
                draw_viewport.size().x() as f32,
                draw_viewport.size().y() as f32,
            ],
            texture_metadata_size: [1024, 1024], // Placeholder
            z_buffer_size: [z_buffer_texture.size.x(), z_buffer_texture.size.y()],
            color_texture_size0: [1024.0, 1024.0], // Placeholder
            mask_texture_size0: [
                mask_viewport.size().x() as f32,
                mask_viewport.size().y() as f32,
            ],
        };

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Globals"),
            contents: bytemuck::cast_slice(&[globals]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Create Bind Groups
        let bind_group_0 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &tile_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                }],
            });

        let sampler = core
            .device
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());
        let mask_storage = core.mask_storage.as_ref().unwrap();
        let mask_texture = core.allocator.get_texture(mask_storage.texture_id);
        let gamma_lut_texture = core.allocator.get_texture(core.gamma_lut_texture_id);

        let bind_group_1 = core
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &tile_pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&metadata_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&z_buffer_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&z_buffer_texture.view),
                    }, // Placeholder for ColorTexture
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&mask_texture.view),
                    }, // Placeholder for DestTexture
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&gamma_lut_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        // 3. Draw
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile Encoder"),
        });
        {
            let dest_texture = core
                .allocator
                .get_texture(core.intermediate_dest_texture_id);

            let clear_color = core.clear_color_for_draw_operation();
            let load_op = if let Some(color) = clear_color {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: color.r() as f64,
                    g: color.g() as f64,
                    b: color.b() as f64,
                    a: color.a() as f64,
                })
            } else {
                wgpu::LoadOp::Load
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tile Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dest_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&tile_pipeline);
            render_pass.set_bind_group(0, &bind_group_0, &[]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);

            let tile_vertex_buffer = core.allocator.get_general_buffer(tile_vertex_buffer_id);
            let quad_vertex_positions_buffer = core
                .allocator
                .get_general_buffer(core.quad_vertex_positions_buffer_id);
            let quad_vertex_indices_buffer = core
                .allocator
                .get_index_buffer(core.quad_vertex_indices_buffer_id);

            render_pass.set_vertex_buffer(0, quad_vertex_positions_buffer.slice(..));
            render_pass.set_vertex_buffer(1, tile_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                quad_vertex_indices_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.set_viewport(
                0.0,
                0.0,
                draw_viewport.size().x() as f32,
                draw_viewport.size().y() as f32,
                0.0,
                1.0,
            );
            render_pass.draw_indexed(0..6, 0, 0..tile_count);
        }

        core.device.queue.submit(Some(encoder.finish()));

        core.stats.total_tile_count += tile_count as usize;
        core.stats.drawcall_count += 1;
        core.finish_timing_draw_call(&mut timer_query);
        core.current_timer
            .as_mut()
            .unwrap()
            .push_query(TimeCategory::Composite, timer_query);
        core.preserve_draw_framebuffer();
    }

    fn copy_alpha_tiles_to_dest_blend_texture(
        &mut self,
        core: &mut RendererCore,
        _tile_count: u32,
        _vertex_buffer_id: GeneralBufferID,
    ) {
        core.stats.drawcall_count += 1;
    }

    fn mask_viewport(&self, core: &RendererCore) -> RectI {
        let page_count = match core.mask_storage {
            Some(ref mask_storage) => mask_storage.allocated_page_count as i32,
            None => 0,
        };
        let height = MASK_TEXTURE_HEIGHT * page_count;
        RectI::new(Vector2I::default(), vec2i(MASK_TEXTURE_WIDTH, height))
    }

    fn tile_transform(&self, core: &RendererCore) -> Transform4F {
        let draw_viewport = core.draw_viewport().size().to_f32();
        let scale = Vector4F::new(2.0 / draw_viewport.x(), -2.0 / draw_viewport.y(), 1.0, 1.0);
        Transform4F::from_scale(scale).translate(Vector4F::new(-1.0, 1.0, 0.0, 1.0))
    }
}

#[derive(Clone)]
pub(crate) struct TileBatchInfoD3D9 {
    pub(crate) tile_count: u32,
    pub(crate) z_buffer_id: GeneralBufferID,
    _tile_vertex_buffer_id: GeneralBufferID,
}

#[derive(Clone)]
struct FillBufferInfoD3D9 {
    fill_buffer_id: GeneralBufferID,
    fill_count: u32,
}

struct TileBufferD3D9 {
    tile_vertex_buffer_id: GeneralBufferID,
}

struct ClipBufferInfo {
    clip_buffer_id: GeneralBufferID,
    clip_count: u32,
}
