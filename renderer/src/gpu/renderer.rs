// pathfinder/renderer/src/gpu/renderer.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The GPU renderer that processes commands necessary to render a scene.

use crate::gpu::blend::ToCompositeCtrl;
#[cfg(feature = "d3d11")]
use crate::gpu::d3d11::renderer::RendererD3D11;
#[cfg(feature = "d3d9")]
use crate::gpu::d3d9::renderer::RendererD3D9;
#[cfg(feature = "ui")]
use crate::gpu::debug::DebugUiPresenter;
use crate::gpu::options::{DestFramebuffer, RendererLevel, RendererMode, RendererOptions};
use crate::gpu::perf::{PendingTimer, RenderStats, RenderTime, TimerQueryCache};
use crate::gpu_data::{
    ColorCombineMode, RenderCommand, TextureLocation, TextureMetadataEntry, TexturePageDescriptor,
    TexturePageId,
};
use crate::options::BoundingQuad;
use crate::tiles::{TILE_HEIGHT, TILE_WIDTH};
use fxhash::FxHashMap;
use half::f16;
use pathfinder_color::{ColorF, ColorU};
use pathfinder_content::effects::{BlendMode, BlurDirection, Filter, PatternFilter};
use pathfinder_content::render_target::RenderTargetId;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::transform3d::Transform4F;
use pathfinder_geometry::util;
use pathfinder_geometry::vector::{vec2f, vec2i, Vector2I};
use pathfinder_gpu::allocator::{BufferTag, GeneralBufferID};
use pathfinder_gpu::allocator::{GpuMemoryAllocator, IndexBufferID, TextureID, TextureTag};
use pathfinder_gpu::Device;
use pathfinder_gpu::RenderTarget;
use pathfinder_gpu::Texture;
use pathfinder_resources::ResourceLoader;
use pathfinder_simd::default::{F32x2, F32x4};
use std::collections::VecDeque;
use std::time::Duration;
use wgpu;
use wgpu::util::DeviceExt;

static QUAD_VERTEX_POSITIONS: [u16; 8] = [0, 0, 1, 0, 1, 1, 0, 1];
static QUAD_VERTEX_INDICES: [u32; 6] = [0, 1, 3, 1, 2, 3];

pub(crate) const MASK_TILES_ACROSS: u32 = 256;
pub(crate) const MASK_TILES_DOWN: u32 = 256;

// 1.0 / sqrt(2*pi)
const SQRT_2_PI_INV: f32 = 0.3989422804014327;

const TEXTURE_METADATA_ENTRIES_PER_ROW: i32 = 128;
const TEXTURE_METADATA_TEXTURE_WIDTH: i32 = TEXTURE_METADATA_ENTRIES_PER_ROW * 10;
const TEXTURE_METADATA_TEXTURE_HEIGHT: i32 = 65536 / TEXTURE_METADATA_ENTRIES_PER_ROW;

pub(crate) const MASK_TEXTURE_WIDTH: i32 = TILE_WIDTH as i32 * MASK_TILES_ACROSS as i32;
pub(crate) const MASK_TEXTURE_HEIGHT: i32 = TILE_HEIGHT as i32 / 4 * MASK_TILES_DOWN as i32;

const COMBINER_CTRL_FILTER_RADIAL_GRADIENT: i32 = 0x1;
const COMBINER_CTRL_FILTER_TEXT: i32 = 0x2;
const COMBINER_CTRL_FILTER_BLUR: i32 = 0x3;
const COMBINER_CTRL_FILTER_COLOR_MATRIX: i32 = 0x4;

const COMBINER_CTRL_COLOR_FILTER_SHIFT: i32 = 4;
const COMBINER_CTRL_COLOR_COMBINE_SHIFT: i32 = 8;
const COMBINER_CTRL_COMPOSITE_SHIFT: i32 = 10;

pub(crate) const TILE_INSTANCE_SIZE: usize = 16;

struct FilterParams {
    p0: F32x4,
    p1: F32x4,
    p2: F32x4,
    p3: F32x4,
    p4: F32x4,
    ctrl: i32,
}

pub struct DebugUiPresenterInfo<'a> {
    pub device: &'a Device,
    pub allocator: &'a mut GpuMemoryAllocator,
    pub debug_ui_presenter: &'a mut DebugUiPresenter,
}

/// The GPU renderer that processes commands necessary to render a scene.
pub struct Renderer {
    pub(crate) core: RendererCore,

    blit_pipeline: wgpu::RenderPipeline,
    clear_pipeline: wgpu::RenderPipeline,
    stencil_pipeline: wgpu::RenderPipeline,
    reprojection_pipeline: wgpu::RenderPipeline,

    #[cfg(feature = "d3d11")]
    d3d11_renderer: RendererD3D11,

    #[cfg(feature = "d3d9")]
    d3d9_renderer: RendererD3D9,

    #[cfg(feature = "debug")]
    current_cpu_build_time: Option<Duration>,
    #[cfg(feature = "debug")]
    pending_timers: VecDeque<PendingTimer>,

    #[cfg(feature = "ui")]
    debug_ui_presenter: Option<DebugUiPresenter>,
    #[cfg(feature = "ui")]
    last_stats: VecDeque<RenderStats>,

    #[cfg(feature = "debug")]
    last_rendering_time: Option<RenderTime>,
}

pub(crate) struct RendererCore {
    pub(crate) device: Device,
    pub(crate) mode: RendererMode,
    pub(crate) allocator: GpuMemoryAllocator,
    pub(crate) options: RendererOptions,
    pub(crate) timer_query_cache: TimerQueryCache,
    pub(crate) quad_vertex_positions_buffer_id: GeneralBufferID,
    pub(crate) quad_vertex_indices_buffer_id: IndexBufferID,
    pub(crate) area_lut_texture_id: TextureID,
    pub(crate) gamma_lut_texture_id: TextureID,
    pub(crate) renderer_flags: RendererFlags,
    pub(crate) mask_storage_flags: MaskStorageFlags,
    pub(crate) stats: RenderStats,
    pub(crate) current_timer: Option<PendingTimer>,
    pub(crate) alpha_tile_count: u32,
    pub(crate) mask_storage: Option<MaskStorage>,
    pub(crate) intermediate_dest_texture_id: TextureID,
    // Texture pages for paint data
    pub(crate) texture_pages: FxHashMap<TexturePageId, TextureID>,
    // Texture metadata texture (stores color/blend info)
    pub(crate) texture_metadata_texture_id: TextureID,
    // Render target stack for off-screen rendering
    pub(crate) render_target_stack: Vec<RenderTargetId>,
    // Mapping from render target ID to texture location
    pub(crate) render_target_textures: FxHashMap<RenderTargetId, TextureLocation>,
}

impl RendererCore {
    pub(crate) fn draw_viewport(&self) -> RectI {
        match self.options.dest {
            DestFramebuffer::Default { viewport, .. } => viewport,
            DestFramebuffer::Other(_) => {
                let texture = self
                    .allocator
                    .get_texture(self.intermediate_dest_texture_id);
                RectI::new(Vector2I::zero(), texture.size)
            }
        }
    }

    pub fn draw_render_target(&self) -> RenderTarget {
        match self.options.dest {
            DestFramebuffer::Default { .. } => RenderTarget::Default,
            DestFramebuffer::Other(ref texture) => RenderTarget::Framebuffer(&texture.view),
        }
    }

    pub fn clear_color_for_draw_operation(&self) -> Option<ColorF> {
        self.options.background_color
    }

    pub fn preserve_draw_framebuffer(&mut self) {
        // ...
    }

    pub fn tile_size(&self) -> Vector2I {
        Vector2I::new(TILE_WIDTH as i32, TILE_HEIGHT as i32)
    }

    pub fn framebuffer_tile_size(&self) -> Vector2I {
        let viewport = self.draw_viewport();
        let size = viewport.size();
        Vector2I::new(
            (size.x() + TILE_WIDTH as i32 - 1) / TILE_WIDTH as i32,
            (size.y() + TILE_HEIGHT as i32 - 1) / TILE_HEIGHT as i32,
        )
    }

    pub(crate) fn mask_texture_format(&self) -> wgpu::TextureFormat {
        match self.mode.level {
            RendererLevel::D3D9 => wgpu::TextureFormat::Rgba16Float,
            RendererLevel::D3D11 => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    pub(crate) fn reallocate_alpha_tile_pages_if_necessary(&mut self, _preserve_contents: bool) {
        let tiles_per_page = MASK_TILES_ACROSS * MASK_TILES_DOWN;
        let needed_page_count = (self.alpha_tile_count + tiles_per_page - 1) / tiles_per_page;
        let needed_page_count = needed_page_count.max(1);

        let current_page_count = self
            .mask_storage
            .as_ref()
            .map(|s| s.allocated_page_count)
            .unwrap_or(0);
        if needed_page_count <= current_page_count {
            return;
        }

        if let Some(storage) = self.mask_storage.take() {
            self.allocator.free_texture(storage.texture_id);
        }

        let mask_size = Vector2I::new(
            MASK_TEXTURE_WIDTH,
            MASK_TEXTURE_HEIGHT * needed_page_count as i32,
        );

        let texture_id = self.allocator.allocate_texture(
            &self.device,
            mask_size,
            self.mask_texture_format(),
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            TextureTag("MaskStorage"),
        );

        self.mask_storage = Some(MaskStorage {
            texture_id,
            allocated_page_count: needed_page_count,
        });

        self.mask_storage_flags
            .insert(MaskStorageFlags::MASK_TEXTURE_IS_DIRTY);
    }

    pub fn finish_timing_draw_call(
        &mut self,
        timer_query: &mut Option<pathfinder_gpu::TimerQuery>,
    ) {
        if let Some(ref mut timer_query) = *timer_query {
            self.device.end_timer_query(timer_query);
        }
    }
}

pub(crate) struct MaskStorage {
    pub(crate) texture_id: TextureID,
    pub(crate) allocated_page_count: u32,
}

bitflags! {
    pub(crate) struct RendererFlags: u8 {
        const USE_DEPTH = 0x01;
    }
}

bitflags! {
    pub(crate) struct MaskStorageFlags: u8 {
        const MASK_TEXTURE_IS_DIRTY = 0x01;
    }
}

impl Renderer {
    /// Creates a new renderer ready to render Pathfinder content.
    ///
    /// Arguments:
    ///
    /// * `device`: The GPU device to render with. This effectively specifies the system GPU API
    ///   Pathfinder will use (OpenGL, Metal, etc.)
    ///
    /// * `resources`: Where Pathfinder should find shaders, lookup tables, and other data.
    ///   This is typically either an `EmbeddedResourceLoader` to use resources included in the
    ///   Pathfinder library or (less commonly) a `FilesystemResourceLoader` to use resources
    ///   stored in a directory on disk.
    ///
    /// * `mode`: Renderer options that can't be changed after the renderer is created. Most
    ///   notably, this specifies the API level (D3D9 or D3D11).
    ///
    /// * `options`: Renderer options that can be changed after the renderer is created. Most
    ///   importantly, this specifies where the output should go (to a window or off-screen).
    pub fn new(
        device: Device,
        resources: &dyn ResourceLoader,
        mode: RendererMode,
        options: RendererOptions,
    ) -> Renderer {
        let mut allocator = GpuMemoryAllocator::new();

        let quad_vertex_positions_buffer_id = allocator.allocate_general_buffer::<u16>(
            &device,
            QUAD_VERTEX_POSITIONS.len() as u64,
            BufferTag("QuadVertexPositions"),
        );
        device.upload_to_buffer(
            &allocator.get_general_buffer(quad_vertex_positions_buffer_id),
            0,
            &QUAD_VERTEX_POSITIONS,
        );

        let quad_vertex_indices_buffer_id = allocator.allocate_index_buffer::<u32>(
            &device,
            QUAD_VERTEX_INDICES.len() as u64,
            BufferTag("QuadVertexIndices"),
        );
        device.upload_to_buffer(
            &allocator.get_index_buffer(quad_vertex_indices_buffer_id),
            0,
            &QUAD_VERTEX_INDICES,
        );

        let area_lut_texture_id = allocator.allocate_texture(
            &device,
            Vector2I::splat(256),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            TextureTag("AreaLUT"),
        );

        let gamma_lut_texture_id = allocator.allocate_texture(
            &device,
            vec2i(256, 8),
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            TextureTag("GammaLUT"),
        );

        device.upload_png_to_texture(
            resources,
            "area-lut",
            allocator.get_texture(area_lut_texture_id),
        );

        device.upload_png_to_texture(
            resources,
            "gamma-lut",
            allocator.get_texture(gamma_lut_texture_id),
        );

        let window_size = options.dest.window_size(&device);
        let intermediate_dest_texture_id = allocator.allocate_texture(
            &device,
            window_size,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            TextureTag("IntermediateDest"),
        );

        let texture_metadata_texture_size = vec2i(
            TEXTURE_METADATA_TEXTURE_WIDTH,
            TEXTURE_METADATA_TEXTURE_HEIGHT,
        );
        let texture_metadata_texture_id = allocator.allocate_texture(
            &device,
            texture_metadata_texture_size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            TextureTag("TextureMetadata"),
        );

        let mut core = RendererCore {
            device: device.clone(),
            mode: mode.clone(),
            allocator,
            options,
            timer_query_cache: TimerQueryCache::new(),
            quad_vertex_positions_buffer_id,
            quad_vertex_indices_buffer_id,
            area_lut_texture_id,
            gamma_lut_texture_id,
            texture_metadata_texture_id,
            renderer_flags: RendererFlags::empty(),
            mask_storage_flags: MaskStorageFlags::empty(),
            stats: RenderStats::default(),
            current_timer: None,
            alpha_tile_count: 0,
            mask_storage: None,
            intermediate_dest_texture_id,
            texture_pages: FxHashMap::default(),
            render_target_stack: Vec::new(),
            render_target_textures: FxHashMap::default(),
        };

        let blit_pipeline = device.create_render_pipeline(resources, "blit", None);
        let clear_pipeline = device.create_render_pipeline(resources, "clear", None);
        let stencil_pipeline = device.create_render_pipeline(resources, "stencil", None);
        let reprojection_pipeline = device.create_render_pipeline(resources, "reproject", None);

        #[cfg(feature = "d3d11")]
        let d3d11_renderer = RendererD3D11::new(&core, resources);

        let mut core_mut = core;
        #[cfg(feature = "d3d9")]
        let d3d9_renderer = RendererD3D9::new(&mut core_mut, resources);

        Renderer {
            core: core_mut,
            blit_pipeline,
            clear_pipeline,
            stencil_pipeline,
            reprojection_pipeline,
            #[cfg(feature = "d3d11")]
            d3d11_renderer,
            #[cfg(feature = "d3d9")]
            d3d9_renderer,
            #[cfg(feature = "debug")]
            current_cpu_build_time: None,
            #[cfg(feature = "debug")]
            pending_timers: VecDeque::new(),
            #[cfg(feature = "ui")]
            debug_ui_presenter: Some(DebugUiPresenter::new(
                &device,
                resources,
                window_size,
                mode.level,
            )),
            #[cfg(feature = "ui")]
            last_stats: VecDeque::new(),
            #[cfg(feature = "debug")]
            last_rendering_time: None,
        }
    }

    pub fn device(&self) -> &Device {
        &self.core.device
    }
    pub fn device_mut(&mut self) -> &mut Device {
        &mut self.core.device
    }

    pub fn options(&self) -> &RendererOptions {
        &self.core.options
    }
    pub fn options_mut(&mut self) -> &mut RendererOptions {
        &mut self.core.options
    }

    pub fn draw_viewport(&self) -> RectI {
        self.core.draw_viewport()
    }

    pub fn quad_vertex_positions_buffer(&self) -> &wgpu::Buffer {
        self.core
            .allocator
            .get_general_buffer(self.core.quad_vertex_positions_buffer_id)
    }

    pub fn quad_vertex_indices_buffer(&self) -> &wgpu::Buffer {
        self.core
            .allocator
            .get_index_buffer(self.core.quad_vertex_indices_buffer_id)
    }

    /// Returns the intermediate destination texture that contains the rendered scene.
    /// This should be blitted to the screen surface during present.
    pub fn intermediate_dest_texture(&self) -> &Texture {
        self.core
            .allocator
            .get_texture(self.core.intermediate_dest_texture_id)
    }

    /// Blit the intermediate destination texture to the given surface texture view.
    /// Uses the blit pipeline (blit.wgsl) to perform the copy via a render pass.
    pub fn blit_to_surface(&self, surface_view: &wgpu::TextureView, surface_size: Vector2I) {
        let device = &self.core.device.device;
        let queue = &self.core.device.queue;

        let intermediate_texture = self.intermediate_dest_texture();
        let intermediate_size = intermediate_texture.size;

        let globals_data = [
            0.0f32,
            0.0f32,
            intermediate_size.x() as f32,
            intermediate_size.y() as f32,
            surface_size.x() as f32,
            surface_size.y() as f32,
            0.0f32,
            0.0f32,
        ];

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blit Globals"),
            contents: bytemuck::cast_slice(&globals_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bg0_layout = self.blit_pipeline.get_bind_group_layout(0);
        let bg1_layout = self.blit_pipeline.get_bind_group_layout(1);
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Globals BG"),
            layout: &bg0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Texture BG"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intermediate_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Blit encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &bg0, &[]);
            render_pass.set_bind_group(1, &bg1, &[]);
            render_pass.draw(0..3, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    #[cfg(feature = "ui")]
    pub fn debug_ui_presenter_mut(&mut self) -> DebugUiPresenterInfo {
        DebugUiPresenterInfo {
            device: &self.core.device,
            allocator: &mut self.core.allocator,
            debug_ui_presenter: self.debug_ui_presenter.as_mut().unwrap(),
        }
    }

    pub fn dest_framebuffer_size_changed(&mut self) {
        // TODO: Update intermediate framebuffer if necessary
    }

    pub fn disable_depth(&mut self) {
        self.core.renderer_flags.remove(RendererFlags::USE_DEPTH);
    }

    pub fn enable_depth(&mut self) {
        self.core.renderer_flags.insert(RendererFlags::USE_DEPTH);
    }

    pub fn reproject_texture(
        &mut self,
        texture: &Texture,
        old_transform: &Transform4F,
        new_transform: &Transform4F,
    ) {
        // TODO: Implement reprojection pass
    }

    pub fn begin_scene(&mut self) {
        self.core.allocator.begin_frame();
        self.core.stats = RenderStats::default();
        self.core.alpha_tile_count = 0;
        self.core
            .mask_storage_flags
            .remove(MaskStorageFlags::MASK_TEXTURE_IS_DIRTY);
        self.core.current_timer = Some(PendingTimer::new());
    }

    pub fn render_command(&mut self, command: &RenderCommand) {
        match command {
            RenderCommand::Start {
                path_count,
                bounding_quad,
                needs_readable_framebuffer,
            } => {
                self.start_rendering(*path_count, *bounding_quad, *needs_readable_framebuffer);
            }
            RenderCommand::AllocateTexturePage {
                page_id,
                descriptor,
            } => {
                self.allocate_texture_page(page_id, descriptor);
            }
            RenderCommand::UploadTexelData { texels, location } => {
                self.upload_texel_data(texels, location);
            }
            RenderCommand::UploadTextureMetadata(metadata) => {
                self.upload_texture_metadata(metadata);
            }
            RenderCommand::DeclareRenderTarget { id, location } => {
                self.core.render_target_textures.insert(*id, *location);
            }
            RenderCommand::PushRenderTarget(render_target_id) => {
                self.push_render_target(render_target_id);
            }
            RenderCommand::PopRenderTarget => {
                self.pop_render_target();
            }
            RenderCommand::Finish { cpu_build_time } => {
                self.core.stats.cpu_build_time = *cpu_build_time;
                self.finish_frame();
            }
            #[cfg(feature = "d3d11")]
            RenderCommand::UploadSceneD3D11 {
                draw_segments,
                clip_segments,
            } => {
                self.d3d11_renderer
                    .upload_scene(&mut self.core, draw_segments, clip_segments);
            }
            #[cfg(feature = "d3d11")]
            RenderCommand::DrawTilesD3D11(batch) => {
                self.d3d11_renderer
                    .prepare_and_draw_tiles(&mut self.core, batch);
            }
            #[cfg(feature = "d3d11")]
            RenderCommand::PrepareClipTilesD3D11(batch_data) => {
                self.d3d11_renderer
                    .prepare_tiles(&mut self.core, batch_data);
            }
            #[cfg(feature = "d3d9")]
            RenderCommand::AddFillsD3D9(fills) => {
                self.d3d9_renderer.add_fills(&mut self.core, fills);
            }
            #[cfg(feature = "d3d9")]
            RenderCommand::FlushFillsD3D9 => {
                self.d3d9_renderer.draw_buffered_fills(&mut self.core);
            }
            #[cfg(feature = "d3d9")]
            RenderCommand::DrawTilesD3D9(batch) => {
                self.d3d9_renderer
                    .upload_and_draw_tiles(&mut self.core, batch);
            }
        }
    }

    fn allocate_texture_page(
        &mut self,
        page_id: &TexturePageId,
        descriptor: &TexturePageDescriptor,
    ) {
        let texture_id = self.core.allocator.allocate_texture(
            &self.core.device,
            descriptor.size,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            TextureTag("TexturePage"),
        );
        self.core.texture_pages.insert(*page_id, texture_id);
    }

    fn upload_texel_data(
        &mut self,
        texels: &std::sync::Arc<Vec<ColorU>>,
        location: &TextureLocation,
    ) {
        let texture_id = self.core.texture_pages.get(&location.page);
        if let Some(texture_id) = texture_id {
            let texture = self.core.allocator.get_texture(*texture_id);
            let pixels: &[u8] = unsafe {
                std::slice::from_raw_parts(texels.as_ptr() as *const u8, texels.len() * 4)
            };
            self.core.device.upload_to_texture(
                texture,
                location.rect,
                pathfinder_gpu::TextureDataRef::U8(pixels),
            );
        }
    }

    fn upload_texture_metadata(&mut self, metadata: &[TextureMetadataEntry]) {
        let padded_texel_size =
            (util::alignup_i32(metadata.len() as i32, TEXTURE_METADATA_ENTRIES_PER_ROW)
                * TEXTURE_METADATA_TEXTURE_WIDTH
                * 4) as usize;
        let mut texels = Vec::with_capacity(padded_texel_size);

        for entry in metadata {
            let base_color = entry.base_color.to_f32();
            let filter_params = self.compute_filter_params(
                &entry.filter,
                entry.blend_mode,
                entry.color_0_combine_mode,
            );
            texels.extend_from_slice(&[
                // 0
                f16::from_f32(entry.color_0_transform.m11()),
                f16::from_f32(entry.color_0_transform.m21()),
                f16::from_f32(entry.color_0_transform.m12()),
                f16::from_f32(entry.color_0_transform.m22()),
                // 1
                f16::from_f32(entry.color_0_transform.m13()),
                f16::from_f32(entry.color_0_transform.m23()),
                f16::default(),
                f16::default(),
                // 2
                f16::from_f32(base_color.r()),
                f16::from_f32(base_color.g()),
                f16::from_f32(base_color.b()),
                f16::from_f32(base_color.a()),
                // 3
                f16::from_f32(filter_params.p0.x()),
                f16::from_f32(filter_params.p0.y()),
                f16::from_f32(filter_params.p0.z()),
                f16::from_f32(filter_params.p0.w()),
                // 4
                f16::from_f32(filter_params.p1.x()),
                f16::from_f32(filter_params.p1.y()),
                f16::from_f32(filter_params.p1.z()),
                f16::from_f32(filter_params.p1.w()),
                // 5
                f16::from_f32(filter_params.p2.x()),
                f16::from_f32(filter_params.p2.y()),
                f16::from_f32(filter_params.p2.z()),
                f16::from_f32(filter_params.p2.w()),
                // 6
                f16::from_f32(filter_params.p3.x()),
                f16::from_f32(filter_params.p3.y()),
                f16::from_f32(filter_params.p3.z()),
                f16::from_f32(filter_params.p3.w()),
                // 7
                f16::from_f32(filter_params.p4.x()),
                f16::from_f32(filter_params.p4.y()),
                f16::from_f32(filter_params.p4.z()),
                f16::from_f32(filter_params.p4.w()),
                // 8
                f16::from_f32(filter_params.ctrl as f32),
                f16::default(),
                f16::default(),
                f16::default(),
                // 9
                f16::default(),
                f16::default(),
                f16::default(),
                f16::default(),
            ]);
        }
        while texels.len() < padded_texel_size {
            texels.push(f16::default())
        }

        let texture_id = self.core.texture_metadata_texture_id;
        let texture = self.core.allocator.get_texture(texture_id);
        let width = TEXTURE_METADATA_TEXTURE_WIDTH;
        let height = texels.len() as i32 / (4 * TEXTURE_METADATA_TEXTURE_WIDTH);

        // Convert f16 to bytes
        let texels_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(texels.as_ptr() as *const u8, texels.len() * 2) };

        self.core.device.upload_to_texture(
            texture,
            RectI::new(Vector2I::default(), vec2i(width, height)),
            pathfinder_gpu::TextureDataRef::F16(&texels),
        );
    }

    fn compute_filter_params(
        &self,
        filter: &Filter,
        blend_mode: BlendMode,
        color_0_combine_mode: ColorCombineMode,
    ) -> FilterParams {
        let mut ctrl = 0;
        ctrl |= blend_mode.to_composite_ctrl() << COMBINER_CTRL_COMPOSITE_SHIFT;
        ctrl |= color_0_combine_mode.to_composite_ctrl() << COMBINER_CTRL_COLOR_COMBINE_SHIFT;

        match *filter {
            Filter::RadialGradient {
                line,
                radii,
                uv_origin,
            } => FilterParams {
                p0: line.from().0.concat_xy_xy(line.vector().0),
                p1: radii.concat_xy_xy(uv_origin.0),
                p2: F32x4::default(),
                p3: F32x4::default(),
                p4: F32x4::default(),
                ctrl: ctrl
                    | (COMBINER_CTRL_FILTER_RADIAL_GRADIENT << COMBINER_CTRL_COLOR_FILTER_SHIFT),
            },
            Filter::PatternFilter(PatternFilter::Blur { sigma, direction }) => {
                let sigma_inv = 1.0 / sigma;
                let gauss_coeff_x = SQRT_2_PI_INV * sigma_inv;
                let gauss_coeff_y = f32::exp(-0.5 * sigma_inv * sigma_inv);
                let gauss_coeff_z = gauss_coeff_y * gauss_coeff_y;

                let src_offset = match direction {
                    BlurDirection::X => vec2f(1.0, 0.0),
                    BlurDirection::Y => vec2f(0.0, 1.0),
                };

                let support = f32::ceil(1.5 * sigma) * 2.0;

                FilterParams {
                    p0: src_offset.0.concat_xy_xy(F32x2::new(support, 0.0)),
                    p1: F32x4::new(gauss_coeff_x, gauss_coeff_y, gauss_coeff_z, 0.0),
                    p2: F32x4::default(),
                    p3: F32x4::default(),
                    p4: F32x4::default(),
                    ctrl: ctrl | (COMBINER_CTRL_FILTER_BLUR << COMBINER_CTRL_COLOR_FILTER_SHIFT),
                }
            }
            Filter::PatternFilter(PatternFilter::Text {
                fg_color,
                bg_color,
                defringing_kernel,
                gamma_correction,
            }) => {
                let mut p2 = fg_color.0;
                p2.set_w(gamma_correction as i32 as f32);

                FilterParams {
                    p0: match defringing_kernel {
                        Some(ref kernel) => F32x4::from_slice(&kernel.0),
                        None => F32x4::default(),
                    },
                    p1: bg_color.0,
                    p2,
                    p3: F32x4::default(),
                    p4: F32x4::default(),
                    ctrl: ctrl | (COMBINER_CTRL_FILTER_TEXT << COMBINER_CTRL_COLOR_FILTER_SHIFT),
                }
            }
            Filter::PatternFilter(PatternFilter::ColorMatrix(matrix)) => {
                let [p0, p1, p2, p3, p4] = matrix.0;
                FilterParams {
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    ctrl: ctrl
                        | (COMBINER_CTRL_FILTER_COLOR_MATRIX << COMBINER_CTRL_COLOR_FILTER_SHIFT),
                }
            }
            Filter::None => FilterParams {
                p0: F32x4::default(),
                p1: F32x4::default(),
                p2: F32x4::default(),
                p3: F32x4::default(),
                p4: F32x4::default(),
                ctrl,
            },
        }
    }

    fn push_render_target(&mut self, render_target_id: &RenderTargetId) {
        self.core.render_target_stack.push(*render_target_id);
    }

    fn pop_render_target(&mut self) {
        self.core.render_target_stack.pop();
    }

    fn finish_frame(&mut self) {
        // The intermediate texture is ready for presentation.
        // The actual blit to screen surface is handled by WindowImpl::present_texture().
        // No additional work needed here.
    }

    pub fn end_scene(&mut self) {
        self.core.stats.gpu_bytes_allocated = self.core.allocator.bytes_allocated();
        self.core.stats.gpu_bytes_committed = self.core.allocator.bytes_committed();

        // match self.level_impl {
        //     #[cfg(feature="d3d9")]
        //     RendererLevel::D3D9(_) => {}
        //     #[cfg(feature="d3d11")]
        //     RendererLevel::D3D11(ref mut d3d11_renderer) => {
        //         d3d11_renderer.end_frame(&mut self.core)
        //     }
        // }

        #[cfg(feature = "debug")]
        {
            if let Some(timer) = self.core.current_timer.take() {
                self.pending_timers.push_back(timer);
            }
            self.current_cpu_build_time = None;
        }

        #[cfg(feature = "ui")]
        {
            self.update_debug_ui();
            if self.core.options.show_debug_ui {
                self.draw_debug_ui();
            }
        }

        self.core.allocator.purge_if_needed();
    }

    fn start_rendering(
        &mut self,
        path_count: usize,
        bounding_quad: BoundingQuad,
        needs_readable_framebuffer: bool,
    ) {
        // match (&self.core.options.dest, self.core.mode.level) {
        //     (&DestFramebuffer::Other(_), _) => {
        //         self.core
        //             .renderer_flags
        //             .remove(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED);
        //     }
        //     (&DestFramebuffer::Default { .. }, RendererLevel::D3D11) => {
        //         self.core
        //             .renderer_flags
        //             .insert(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED);
        //     }
        //     _ => {
        //         self.core
        //             .renderer_flags
        //             .set(RendererFlags::INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED,
        //                  needs_readable_framebuffer);
        //     }
        // }
        //
        // if self.core.renderer_flags.contains(RendererFlags::USE_DEPTH) {
        //     self.draw_stencil(&bounding_quad);
        // }

        self.core.stats.path_count = path_count;

        // self.core.render_targets.clear();
    }

    #[cfg(feature = "ui")]
    fn update_debug_ui(&mut self) {
        self.last_stats.push_back(self.core.stats);
        self.shift_rendering_time();

        if !self.core.options.show_debug_ui || self.debug_ui_presenter.is_none() {
            return;
        }

        if let Some(last_rendering_time) = self.last_rendering_time {
            self.debug_ui_presenter
                .as_mut()
                .unwrap()
                .add_sample(self.last_stats.pop_front().unwrap(), last_rendering_time);
        }
    }

    #[cfg(feature = "debug")]
    fn shift_rendering_time(&mut self) {
        if let Some(mut pending_timer) = self.pending_timers.pop_front() {
            for old_query in pending_timer.poll(&self.core.device) {
                self.core.timer_query_cache.free(old_query);
            }
            if let Some(render_time) = pending_timer.total_time() {
                self.last_rendering_time = Some(render_time);
                return;
            }
            self.pending_timers.push_front(pending_timer);
        }
        self.last_rendering_time = None;
    }

    #[cfg(feature = "ui")]
    fn draw_debug_ui(&mut self) {
        if let Some(ref mut debug_ui_presenter) = self.debug_ui_presenter {
            let window_size = self.core.options.dest.window_size(&self.core.device);
            debug_ui_presenter.set_framebuffer_size(window_size);
            debug_ui_presenter.draw(&self.core.device, &mut self.core.allocator);
        }
    }

    pub fn mode(&self) -> RendererMode {
        RendererMode::default_for_device(&self.core.device)
    }

    pub fn draw_render_target(&self) -> RenderTarget {
        self.core.draw_render_target()
    }
}
