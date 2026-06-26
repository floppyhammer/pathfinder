// pathfinder/gpu/src/lib.rs
//
// Copyright © 2024 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Concrete GPU abstractions based on wgpu.

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;

pub mod allocator;

use half::f16;
use image::ImageFormat;
use pathfinder_color::ColorF;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::transform3d::Transform4F;
use pathfinder_geometry::vector::{vec2i, Vector2I};
use pathfinder_resources::ResourceLoader;
use pathfinder_simd::default::{F32x2, F32x4, I32x2};
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Duration;
use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct Device {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub device_name: String,
    pub backend_name: String,
    pub(crate) encoder: Arc<RefCell<Option<wgpu::CommandEncoder>>>,
}

pub struct TimerQuery {
    start_time: Option<std::time::Instant>,
    end_time: Option<std::time::Instant>,
}

impl TimerQuery {
    fn new() -> TimerQuery {
        TimerQuery {
            start_time: None,
            end_time: None,
        }
    }

    fn begin(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }

    fn end(&mut self) {
        self.end_time = Some(std::time::Instant::now());
    }

    fn elapsed(&self) -> Option<Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: Vector2I,
    pub format: wgpu::TextureFormat,
}

impl Texture {
    pub fn create_view(&self, desc: &wgpu::TextureViewDescriptor) -> wgpu::TextureView {
        self.texture.create_view(desc)
    }

    pub fn create_default_view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }
}

pub enum RenderTarget<'a> {
    Default,
    Framebuffer(&'a wgpu::TextureView),
}

impl Device {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_name: String,
        backend_name: String,
    ) -> Device {
        Device {
            device,
            queue,
            device_name,
            backend_name,
            encoder: Arc::new(RefCell::new(None)),
        }
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn create_texture(
        &self,
        format: wgpu::TextureFormat,
        size: Vector2I,
        usage: wgpu::TextureUsages,
    ) -> Texture {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.x() as u32,
                height: size.y() as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Texture {
            texture,
            view,
            size,
            format,
        }
    }

    pub fn upload_png_to_texture(
        &self,
        resources: &dyn ResourceLoader,
        name: &str,
        texture: &Texture,
    ) {
        let data = resources.slurp(&format!("textures/{}.png", name)).unwrap();
        let image = image::load_from_memory_with_format(&data, ImageFormat::Png).unwrap();
        let format = texture.format;
        match format {
            wgpu::TextureFormat::R8Unorm => {
                let image = image.to_luma8();
                let size = vec2i(image.width() as i32, image.height() as i32);
                let rect = RectI::new(Vector2I::default(), size);
                self.upload_to_texture(&texture, rect, TextureDataRef::U8(&image))
            }
            wgpu::TextureFormat::Rgba8Unorm => {
                let image = image.to_rgba8();
                let size = vec2i(image.width() as i32, image.height() as i32);
                let rect = RectI::new(Vector2I::default(), size);
                self.upload_to_texture(&texture, rect, TextureDataRef::U8(&image))
            }
            _ => unimplemented!(),
        }
    }

    pub fn upload_to_buffer<T>(&self, buffer: &wgpu::Buffer, position: usize, data: &[T]) {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        self.queue.write_buffer(buffer, position as u64, bytes);
    }

    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn create_buffer_with_data<T: bytemuck::Pod>(
        &self,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    pub fn texture_size(&self, texture: &Texture) -> Vector2I {
        texture.size
    }

    pub fn upload_to_texture(&self, texture: &Texture, rect: RectI, data: TextureDataRef) {
        let bytes = match data {
            TextureDataRef::U8(d) => d,
            TextureDataRef::F16(d) => unsafe {
                std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 2)
            },
            TextureDataRef::F32(d) => unsafe {
                std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4)
            },
        };

        let block_size = texture.format.block_copy_size(None).unwrap_or(4);

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.origin().x() as u32,
                    y: rect.origin().y() as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(rect.size().x() as u32 * block_size),
                rows_per_image: Some(rect.size().y() as u32),
            },
            wgpu::Extent3d {
                width: rect.size().x() as u32,
                height: rect.size().y() as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn begin_commands(&self) {
        let mut encoder = self.encoder.borrow_mut();
        if encoder.is_none() {
            *encoder = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            );
        }
    }

    pub fn end_commands(&self) {
        let mut encoder = self.encoder.borrow_mut();
        if let Some(e) = encoder.take() {
            self.queue.submit(std::iter::once(e.finish()));
        }
    }

    pub fn draw_instanced(
        &self,
        target: &RenderTarget,
        pipeline: &wgpu::RenderPipeline,
        bind_groups: &[wgpu::BindGroup],
        vertex_buffers: &[(&wgpu::Buffer, u64)],
        index_buffer: Option<(&wgpu::Buffer, u64, wgpu::IndexFormat)>,
        count: u32,
        instances: u32,
        clear_color: Option<ColorF>,
    ) {
        let mut encoder_borrow = self.encoder.borrow_mut();
        if encoder_borrow.is_none() {
            *encoder_borrow = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            );
        }
        let encoder = encoder_borrow.as_mut().unwrap();

        let view = match target {
            RenderTarget::Framebuffer(v) => v,
            RenderTarget::Default => panic!("Default target not supported"),
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
            depth_stencil_attachment: None, // TODO: Support depth
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            render_pass.set_bind_group(i as u32, bg, &[]);
        }

        for (i, (buffer, offset)) in vertex_buffers.iter().enumerate() {
            render_pass.set_vertex_buffer(i as u32, buffer.slice(*offset..));
        }

        if let Some((buffer, offset, format)) = index_buffer {
            render_pass.set_index_buffer(buffer.slice(offset..), format);
            render_pass.draw_indexed(0..count, 0, 0..instances);
        } else {
            render_pass.draw(0..count, 0..instances);
        }
    }

    pub fn blit_texture(
        &self,
        source: &Texture,
        source_rect: RectI,
        dest: &Texture,
        dest_rect: RectI,
    ) {
        let mut encoder_borrow = self.encoder.borrow_mut();
        if encoder_borrow.is_none() {
            *encoder_borrow = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            );
        }
        let encoder = encoder_borrow.as_mut().unwrap();

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: source_rect.origin().x() as u32,
                    y: rect_y(source_rect, source.size),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dest.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: dest_rect.origin().x() as u32,
                    y: rect_y(dest_rect, dest.size),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: source_rect.size().x() as u32,
                height: source_rect.size().y() as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_groups: &[wgpu::BindGroup],
        num_groups: (u32, u32, u32),
    ) {
        let mut encoder_borrow = self.encoder.borrow_mut();
        if encoder_borrow.is_none() {
            *encoder_borrow = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            );
        }
        let encoder = encoder_borrow.as_mut().unwrap();

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, bg, &[]);
        }
        compute_pass.dispatch_workgroups(num_groups.0, num_groups.1, num_groups.2);
    }

    pub fn create_timer_query(&self) -> TimerQuery {
        TimerQuery::new()
    }

    pub fn begin_timer_query(&self, query: &mut TimerQuery) {
        query.begin();
    }

    pub fn end_timer_query(&self, query: &mut TimerQuery) {
        query.end();
    }

    pub fn try_recv_timer_query(&self, query: &TimerQuery) -> Option<Duration> {
        query.elapsed()
    }

    pub fn read_pixels(&self, _target: &RenderTarget, _viewport: RectI) -> () {
        ()
    }

    pub fn recv_texture_data(&self, _receiver: &()) -> TextureData {
        TextureData::U8(vec![])
    }

    pub fn create_render_pipeline(
        &self,
        resources: &dyn ResourceLoader,
        name: &str,
        extra: Option<&str>,
    ) -> wgpu::RenderPipeline {
        let path = format!("shaders/{}.wgsl", name);
        let source = resources.slurp(&path).expect("Failed to load shader");

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(String::from_utf8_lossy(&source).into()),
            });

        if name.contains("blit") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Blit Globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let bgl1 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Blit Texture"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Blit Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

            return pipeline;
        }

        if name.contains("clear") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Clear Globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Clear Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("stencil") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Stencil Globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Stencil Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("reproject") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Reproject Globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let bgl1 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Reproject Texture"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Reproject Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("demo_ground") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Demo Ground BG Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Demo Ground Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("d3d9/fill") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("D3D9 Fill Globals"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D9 Fill Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[
                            wgpu::VertexBufferLayout {
                                array_stride: 4,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                            },
                            wgpu::VertexBufferLayout {
                                array_stride: 12,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &wgpu::vertex_attr_array![1 => Uint16x4, 2 => Uint32],
                            },
                        ],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("d3d9/tile") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("D3D9 Tile Globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let bgl1 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("D3D9 Tile Textures"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D9 Tile Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[
                            // Buffer 0: Vertex Step Mode
                            wgpu::VertexBufferLayout {
                                array_stride: 4,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                            },
                            // Buffer 1: Instance Step Mode
                            wgpu::VertexBufferLayout {
                                array_stride: 16,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &[
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Sint16x2,
                                        offset: 0,
                                        shader_location: 1,
                                    },
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Uint8x4,
                                        offset: 4,
                                        shader_location: 2,
                                    },
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Sint32,
                                        offset: 8,
                                        shader_location: 3,
                                    },
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Sint8x2,
                                        offset: 12,
                                        shader_location: 4,
                                    },
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Uint16,
                                        offset: 14,
                                        shader_location: 5,
                                    },
                                ],
                            },
                        ],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        // if name.contains("d3d9/tile_clip") || name.contains("d3d9/tile_copy") {
        //     let bgl0 = self
        //         .device
        //         .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //             label: Some("D3D9 Clip/Copy Globals"),
        //             entries: &[wgpu::BindGroupLayoutEntry {
        //                 binding: 0,
        //                 visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        //                 ty: wgpu::BindingType::Buffer {
        //                     ty: wgpu::BufferBindingType::Uniform,
        //                     has_dynamic_offset: false,
        //                     min_binding_size: None,
        //                 },
        //                 count: None,
        //             }],
        //         });
        //     let bgl1 = self
        //         .device
        //         .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //             label: Some("D3D9 Clip/Copy Textures"),
        //             entries: &[
        //                 wgpu::BindGroupLayoutEntry {
        //                     binding: 0,
        //                     visibility: wgpu::ShaderStages::FRAGMENT,
        //                     ty: wgpu::BindingType::Texture {
        //                         sample_type: wgpu::TextureSampleType::Float { filterable: true },
        //                         view_dimension: wgpu::TextureViewDimension::D2,
        //                         multisampled: false,
        //                     },
        //                     count: None,
        //                 },
        //                 wgpu::BindGroupLayoutEntry {
        //                     binding: 1,
        //                     visibility: wgpu::ShaderStages::FRAGMENT,
        //                     ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        //                     count: None,
        //                 },
        //             ],
        //         });
        //     let pipeline_layout =
        //         self.device
        //             .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //                 label: Some("D3D9 Clip/Copy Layout"),
        //                 bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
        //                 immediate_size: 0,
        //             });
        //     let pipeline = self
        //         .device
        //         .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        //             label: Some(name),
        //             layout: Some(&pipeline_layout),
        //             vertex: wgpu::VertexState {
        //                 module: &module,
        //                 entry_point: Some("vs_main"),
        //                 buffers: &[wgpu::VertexBufferLayout {
        //                     array_stride: 4,
        //                     step_mode: wgpu::VertexStepMode::Vertex,
        //                     attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
        //                 }],
        //                 compilation_options: Default::default(),
        //             },
        //             fragment: Some(wgpu::FragmentState {
        //                 module: &module,
        //                 entry_point: Some("fs_main"),
        //                 targets: &[Some(wgpu::ColorTargetState {
        //                     format: wgpu::TextureFormat::Rgba8Unorm,
        //                     blend: None,
        //                     write_mask: wgpu::ColorWrites::ALL,
        //                 })],
        //                 compilation_options: Default::default(),
        //             }),
        //             primitive: wgpu::PrimitiveState::default(),
        //             depth_stencil: None,
        //             multisample: wgpu::MultisampleState::default(),
        //             multiview_mask: None,
        //             cache: None,
        //         });
        //     return pipeline;
        // }

        if name.contains("d3d11/tile") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("D3D11 Tile Globals"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let bgl1 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("D3D11 Tile Textures"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Tile Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("debug/solid") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Debug Solid BG Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Debug Solid Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });

            let mut topology = wgpu::PrimitiveTopology::TriangleList;
            if let Some(ex) = extra {
                if ex.contains("outline") {
                    topology = wgpu::PrimitiveTopology::LineList
                }
            }

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        if name.contains("debug/texture") {
            let bgl0 = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Debug Texture Globals"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Debug Texture Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 16,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
            return pipeline;
        }

        unimplemented!("Shader {} not implemented in wgpu backend", name);
    }

    pub fn create_compute_pipeline(
        &self,
        resources: &dyn ResourceLoader,
        name: &str,
    ) -> wgpu::ComputePipeline {
        let path = format!("shaders/{}.wgsl", name);
        let source = resources.slurp(&path).expect("Failed to load shader");
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(String::from_utf8_lossy(&source).into()),
            });

        let (bind_group_layouts, pipeline_layout) = match name {
            "d3d11/bound" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Bound Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Bound Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Bound Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/dice" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Dice Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Dice Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Dice Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/bin" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Bin Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Bin Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Bin Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/propagate" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Propagate Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Propagate Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Propagate Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/sort" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Sort Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Sort Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Sort Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/fill" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Fill Globals"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::ReadWrite,
                                    format: wgpu::TextureFormat::Rgba8Unorm,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Fill Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Fill Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1], pl)
            }
            "d3d11/tile" => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Tile Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let bgl1 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Tile Dest"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::ReadWrite,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        }],
                    });
                let bgl2 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Tile Textures"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    });
                let bgl3 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("D3D11 Tile Storage"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("D3D11 Tile Layout"),
                        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2), Some(&bgl3)],
                        immediate_size: 0,
                    });
                (vec![bgl0, bgl1, bgl2, bgl3], pl)
            }
            _ => {
                let bgl0 = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Compute Globals"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Compute Layout"),
                        bind_group_layouts: &[Some(&bgl0)],
                        immediate_size: 0,
                    });
                (vec![bgl0], pl)
            }
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        pipeline
    }
}

#[derive(Clone, Copy, Debug)]
pub enum UniformData {
    Float(f32),
    IVec2(I32x2),
    Int(i32),
    Mat4([F32x4; 4]),
    Vec2(F32x2),
    Vec4(F32x4),
}

impl UniformData {
    pub fn from_transform_3d(transform: &Transform4F) -> UniformData {
        UniformData::Mat4([transform.c0, transform.c1, transform.c2, transform.c3])
    }
}

#[derive(Clone, Debug)]
pub enum TextureData {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

#[derive(Clone, Copy, Debug)]
pub enum TextureDataRef<'a> {
    U8(&'a [u8]),
    F16(&'a [f16]),
    F32(&'a [f32]),
}

bitflags! {
    pub struct TextureSamplingFlags: u8 {
        const REPEAT_U    = 0x01;
        const REPEAT_V    = 0x02;
        const NEAREST_MIN = 0x04;
        const NEAREST_MAG = 0x08;
    }
}

fn rect_y(rect: RectI, texture_size: Vector2I) -> u32 {
    (texture_size.y() - rect.origin().y() - rect.size().y()) as u32
}
