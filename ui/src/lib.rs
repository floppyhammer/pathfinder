// pathfinder/ui/src/lib.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A minimal immediate mode UI, for debugging.
//!
//! This can be used in your own applications as an ultra-minimal lightweight
//! alternative to dear imgui, Conrod, etc.

#[macro_use]
extern crate serde_derive;

use hashbrown::HashMap;
use pathfinder_color::ColorU;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::{vec2i, Vector2F, Vector2I};
use pathfinder_gpu::allocator::{BufferTag, GpuMemoryAllocator};
use pathfinder_gpu::{Device, Texture, TextureDataRef, UniformData, RenderTarget};
use pathfinder_resources::ResourceLoader;
use pathfinder_simd::default::F32x4;
use serde_json;
use std::mem;

pub const PADDING: i32 = 12;

pub const LINE_HEIGHT: i32 = 42;
pub const FONT_ASCENT: i32 = 28;

pub const BUTTON_WIDTH: i32 = PADDING * 2 + ICON_SIZE;
pub const BUTTON_HEIGHT: i32 = PADDING * 2 + ICON_SIZE;
pub const BUTTON_TEXT_OFFSET: i32 = PADDING + 36;

pub const TOOLTIP_HEIGHT: i32 = FONT_ASCENT + PADDING * 2;

const DEBUG_TEXTURE_VERTEX_SIZE: usize = 8;
const DEBUG_SOLID_VERTEX_SIZE:   usize = 4;

const ICON_SIZE: i32 = 48;

const SEGMENT_SIZE: i32 = 96;

pub static TEXT_COLOR: ColorU = ColorU {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};
pub static WINDOW_COLOR: ColorU = ColorU {
    r: 0,
    g: 0,
    b: 0,
    a: 255 - 90,
};

static BUTTON_ICON_COLOR: ColorU = ColorU {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};
static OUTLINE_COLOR: ColorU = ColorU {
    r: 255,
    g: 255,
    b: 255,
    a: 192,
};

static INVERTED_TEXT_COLOR: ColorU = ColorU {
    r: 0,
    g: 0,
    b: 0,
    a: 255,
};

static FONT_JSON_VIRTUAL_PATH: &'static str = "debug-fonts/regular.json";
static FONT_PNG_NAME: &'static str = "debug-font";

static CORNER_FILL_PNG_NAME: &'static str = "debug-corner-fill";
static CORNER_OUTLINE_PNG_NAME: &'static str = "debug-corner-outline";

static QUAD_INDICES: [u32; 6] = [0, 1, 3, 1, 2, 3];
static RECT_LINE_INDICES: [u32; 8] = [0, 1, 1, 2, 2, 3, 3, 0];
static OUTLINE_RECT_LINE_INDICES: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

pub struct UIPresenter {
    pub event_queue: UIEventQueue,
    pub mouse_position: Vector2F,

    framebuffer_size: Vector2I,

    texture_pipeline: wgpu::RenderPipeline,
    solid_filled_pipeline: wgpu::RenderPipeline,
    solid_outline_pipeline: wgpu::RenderPipeline,
    font: DebugFont,

    font_texture: Texture,
    corner_fill_texture: Texture,
    corner_outline_texture: Texture,

    render_target: Option<wgpu::TextureView>,
    
    cached_sampler: wgpu::Sampler,
    cached_transform_buffer: wgpu::Buffer,
}

impl UIPresenter {
    pub fn new(
        device: &Device,
        resources: &dyn ResourceLoader,
        framebuffer_size: Vector2I,
    ) -> UIPresenter {
        let texture_pipeline = device.create_render_pipeline(resources, "debug/texture", None);
        let font = DebugFont::load(resources);

        let solid_filled_pipeline = device.create_render_pipeline(resources, "debug/solid", None);
        let solid_outline_pipeline = device.create_render_pipeline(resources, "debug/solid", Some("outline"));

        // create_texture_from_png was likely a method on Device or an extension.
        // We'll assume for now it's still available or we'll need to adapt.
        let font_data = resources.slurp(&format!("textures/{}.png", FONT_PNG_NAME)).unwrap();
        let font_image = image::load_from_memory_with_format(&font_data, image::ImageFormat::Png).unwrap();
        let font_image = font_image.to_luma8();
        let font_size = vec2i(font_image.width() as i32, font_image.height() as i32);
        let font_texture = device.create_texture(wgpu::TextureFormat::R8Unorm, font_size, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let font_rect = RectI::new(Vector2I::default(), font_size);
        device.upload_to_texture(&font_texture, font_rect, TextureDataRef::U8(&font_image));
        
        let corner_fill_data = resources.slurp(&format!("textures/{}.png", CORNER_FILL_PNG_NAME)).unwrap();
        let corner_fill_image = image::load_from_memory_with_format(&corner_fill_data, image::ImageFormat::Png).unwrap();
        let corner_fill_image = corner_fill_image.to_luma8();
        let corner_fill_size = vec2i(corner_fill_image.width() as i32, corner_fill_image.height() as i32);
        let corner_fill_texture = device.create_texture(wgpu::TextureFormat::R8Unorm, corner_fill_size, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let corner_fill_rect = RectI::new(Vector2I::default(), corner_fill_size);
        device.upload_to_texture(&corner_fill_texture, corner_fill_rect, TextureDataRef::U8(&corner_fill_image));

        let corner_outline_data = resources.slurp(&format!("textures/{}.png", CORNER_OUTLINE_PNG_NAME)).unwrap();
        let corner_outline_image = image::load_from_memory_with_format(&corner_outline_data, image::ImageFormat::Png).unwrap();
        let corner_outline_image = corner_outline_image.to_luma8();
        let corner_outline_size = vec2i(corner_outline_image.width() as i32, corner_outline_image.height() as i32);
        let corner_outline_texture = device.create_texture(wgpu::TextureFormat::R8Unorm, corner_outline_size, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let corner_outline_rect = RectI::new(Vector2I::default(), corner_outline_size);
        device.upload_to_texture(&corner_outline_texture, corner_outline_rect, TextureDataRef::U8(&corner_outline_image));

        let cached_sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let transform = [
            2.0 / framebuffer_size.x() as f32, 0.0, 0.0, 0.0,
            0.0, -2.0 / framebuffer_size.y() as f32, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        ];
        let cached_transform_buffer = device.create_buffer_with_data(
            &transform,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        UIPresenter {
            event_queue: UIEventQueue::new(),
            mouse_position: Vector2F::zero(),

            framebuffer_size,

            texture_pipeline,
            solid_filled_pipeline,
            solid_outline_pipeline,
            font,

            font_texture,
            corner_fill_texture,
            corner_outline_texture,

            render_target: None,
            
            cached_sampler,
            cached_transform_buffer,
        }
    }

    pub fn set_render_target(&mut self, render_target: Option<wgpu::TextureView>) {
        self.render_target = render_target;
    }

    pub fn framebuffer_size(&self) -> Vector2I {
        self.framebuffer_size
    }

    pub fn set_framebuffer_size(&mut self, window_size: Vector2I) {
        self.framebuffer_size = window_size;
    }

    pub fn draw_solid_rect(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        rect: RectI,
        color: ColorU,
    ) {
        self.draw_rect(device, allocator, rect, color, true);
    }

    pub fn draw_rect_outline(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        rect: RectI,
        color: ColorU,
    ) {
        self.draw_rect(device, allocator, rect, color, false);
    }

    fn draw_rect(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        rect: RectI,
        color: ColorU,
        filled: bool,
    ) {
        let vertex_data = [
            DebugSolidVertex::new(rect.origin()),
            DebugSolidVertex::new(rect.upper_right()),
            DebugSolidVertex::new(rect.lower_right()),
            DebugSolidVertex::new(rect.lower_left()),
        ];

        if filled {
            self.draw_solid_rects_with_vertex_data(
                device,
                allocator,
                &vertex_data,
                &QUAD_INDICES,
                color,
                true,
            );
        } else {
            self.draw_solid_rects_with_vertex_data(
                device,
                allocator,
                &vertex_data,
                &RECT_LINE_INDICES,
                color,
                false,
            );
        }
    }

    fn draw_solid_rects_with_vertex_data(
        &self,
        device: &Device,
        _allocator: &mut GpuMemoryAllocator,
        vertex_data: &[DebugSolidVertex],
        index_data: &[u32],
        color: ColorU,
        filled: bool,
    ) {
        // 直接创建独立的物理 Buffer，避免使用 pool 中的旧 Buffer 导致异步冲突
        let vertex_buffer = device.create_buffer_with_data(vertex_data, wgpu::BufferUsages::VERTEX);
        let index_buffer = device.create_buffer_with_data(index_data, wgpu::BufferUsages::INDEX);

        let pipeline = if filled {
            &self.solid_filled_pipeline
        } else {
            &self.solid_outline_pipeline
        };

        let color_uniform = [
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ];

        let color_buffer = device.create_buffer_with_data(
            &color_uniform,
            wgpu::BufferUsages::UNIFORM,
        );

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Solid BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(self.cached_transform_buffer.slice(..).into()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(color_buffer.slice(..).into()),
                },
            ],
        });

        if let Some(ref render_target) = self.render_target {
            device.draw_instanced(
                &RenderTarget::Framebuffer(render_target),
                pipeline,
                &[bind_group],
                &[(&vertex_buffer, 0)],
                Some((&index_buffer, 0, wgpu::IndexFormat::Uint32)),
                index_data.len() as u32,
                1,
                None,
            );
        }
    }

    pub fn draw_text(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        string: &str,
        origin: Vector2I,
        invert: bool,
    ) {
        let font_texture_size = device.texture_size(&self.font_texture);
        let font_texture_size_f = font_texture_size.to_f32();
        
        let mut next = origin;
        let char_count = string.chars().count();
        let mut vertex_data = Vec::with_capacity(char_count * 4);
        let mut index_data = Vec::with_capacity(char_count * 6);
        for mut character in string.chars() {
            if !self.font.characters.contains_key(&character) {
                character = '?';
            }

            let info = &self.font.characters[&character];
            let position_rect = RectI::new(
                vec2i(next.x() - info.origin_x, next.y() - info.origin_y),
                vec2i(info.width as i32, info.height as i32),
            );
            let tex_coord_rect = RectI::new(vec2i(info.x, info.y), vec2i(info.width, info.height));
            let first_vertex_index = vertex_data.len();
            vertex_data.extend_from_slice(&[
                DebugTextureVertex::new(position_rect.origin(), tex_coord_rect.origin().to_f32() / font_texture_size_f),
                DebugTextureVertex::new(position_rect.upper_right(), tex_coord_rect.upper_right().to_f32() / font_texture_size_f),
                DebugTextureVertex::new(position_rect.lower_right(), tex_coord_rect.lower_right().to_f32() / font_texture_size_f),
                DebugTextureVertex::new(position_rect.lower_left(), tex_coord_rect.lower_left().to_f32() / font_texture_size_f),
            ]);
            index_data.extend(QUAD_INDICES.iter().map(|&i| i + first_vertex_index as u32));

            let next_x = next.x() + info.advance;
            next.set_x(next_x);
        }

        let color = if invert {
            INVERTED_TEXT_COLOR
        } else {
            TEXT_COLOR
        };
        self.draw_texture_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            &index_data,
            &self.font_texture,
            color,
        );
    }

    pub fn draw_texture(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        origin: Vector2I,
        texture: &Texture,
        color: ColorU,
    ) {
        let texture_size = device.texture_size(&texture);
        let position_rect = RectI::new(origin, texture_size);
        let tex_coord_rect = RectI::new(Vector2I::default(), texture_size);
        let texture_size_f = texture_size.to_f32();
        let vertex_data = [
            DebugTextureVertex::new(
                position_rect.origin(),
                tex_coord_rect.origin().to_f32() / texture_size_f,
            ),
            DebugTextureVertex::new(
                position_rect.upper_right(),
                tex_coord_rect.upper_right().to_f32() / texture_size_f,
            ),
            DebugTextureVertex::new(
                position_rect.lower_right(),
                tex_coord_rect.lower_right().to_f32() / texture_size_f,
            ),
            DebugTextureVertex::new(
                position_rect.lower_left(),
                tex_coord_rect.lower_left().to_f32() / texture_size_f,
            ),
        ];

        self.draw_texture_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            &QUAD_INDICES,
            texture,
            color,
        );
    }

    pub fn measure_text(&self, string: &str) -> i32 {
        let mut next = 0;
        for mut character in string.chars() {
            if !self.font.characters.contains_key(&character) {
                character = '?';
            }

            let info = &self.font.characters[&character];
            next += info.advance;
        }
        next
    }

    #[inline]
    pub fn measure_segmented_control(&self, segment_count: u8) -> i32 {
        SEGMENT_SIZE * segment_count as i32 + (segment_count - 1) as i32
    }

    pub fn draw_solid_rounded_rect(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        rect: RectI,
        color: ColorU,
    ) {
        let corner_texture = self.corner_texture(true);
        let corner_rects = CornerRects::new(device, rect, corner_texture);
        self.draw_rounded_rect_corners(device, allocator, color, corner_texture, &corner_rects);

        let solid_rect_mid = RectI::from_points(
            corner_rects.upper_left.upper_right(),
            corner_rects.lower_right.lower_left(),
        );
        let solid_rect_left = RectI::from_points(
            corner_rects.upper_left.lower_left(),
            corner_rects.lower_left.upper_right(),
        );
        let solid_rect_right = RectI::from_points(
            corner_rects.upper_right.lower_left(),
            corner_rects.lower_right.upper_right(),
        );
        let vertex_data = vec![
            DebugSolidVertex::new(solid_rect_mid.origin()),
            DebugSolidVertex::new(solid_rect_mid.upper_right()),
            DebugSolidVertex::new(solid_rect_mid.lower_right()),
            DebugSolidVertex::new(solid_rect_mid.lower_left()),
            DebugSolidVertex::new(solid_rect_left.origin()),
            DebugSolidVertex::new(solid_rect_left.upper_right()),
            DebugSolidVertex::new(solid_rect_left.lower_right()),
            DebugSolidVertex::new(solid_rect_left.lower_left()),
            DebugSolidVertex::new(solid_rect_right.origin()),
            DebugSolidVertex::new(solid_rect_right.upper_right()),
            DebugSolidVertex::new(solid_rect_right.lower_right()),
            DebugSolidVertex::new(solid_rect_right.lower_left()),
        ];

        let mut index_data = Vec::with_capacity(18);
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 0));
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 4));
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 8));

        self.draw_solid_rects_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            &index_data[0..18],
            color,
            true,
        );
    }

    pub fn draw_rounded_rect_outline(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        rect: RectI,
        color: ColorU,
    ) {
        let corner_texture = self.corner_texture(false);
        let corner_rects = CornerRects::new(device, rect, corner_texture);
        self.draw_rounded_rect_corners(device, allocator, color, corner_texture, &corner_rects);

        let vertex_data = vec![
            DebugSolidVertex::new(corner_rects.upper_left.upper_right()),
            DebugSolidVertex::new(corner_rects.upper_right.origin()),
            DebugSolidVertex::new(corner_rects.upper_right.lower_right()),
            DebugSolidVertex::new(corner_rects.lower_right.upper_right()),
            DebugSolidVertex::new(corner_rects.lower_left.lower_right()),
            DebugSolidVertex::new(corner_rects.lower_right.lower_left()),
            DebugSolidVertex::new(corner_rects.upper_left.lower_left()),
            DebugSolidVertex::new(corner_rects.lower_left.origin()),
        ];

        let index_data = &OUTLINE_RECT_LINE_INDICES;
        self.draw_solid_rects_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            index_data,
            color,
            false,
        );
    }

    // TODO(pcwalton): `LineSegment2I`.
    fn draw_line(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        from: Vector2I,
        to: Vector2I,
        color: ColorU,
    ) {
        let line_width = 1;
        let half_width = line_width / 2;
        
        let vertex_data = vec![
            DebugSolidVertex::new(Vector2I::new(from.x() - half_width, from.y())),
            DebugSolidVertex::new(Vector2I::new(to.x() - half_width, to.y())),
            DebugSolidVertex::new(Vector2I::new(to.x() + half_width, to.y())),
            DebugSolidVertex::new(Vector2I::new(from.x() + half_width, from.y())),
        ];
        
        self.draw_solid_rects_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            &QUAD_INDICES,
            color,
            true,
        );
    }

    fn draw_rounded_rect_corners(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        color: ColorU,
        texture: &Texture,
        corner_rects: &CornerRects,
    ) {
        let corner_size = device.texture_size(&texture);
        let tex_coord_rect = RectI::new(Vector2I::default(), corner_size);
        let corner_size_f = corner_size.to_f32();

        let vertex_data = vec![
            DebugTextureVertex::new(
                corner_rects.upper_left.origin(),
                tex_coord_rect.origin().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_left.upper_right(),
                tex_coord_rect.upper_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_left.lower_right(),
                tex_coord_rect.lower_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_left.lower_left(),
                tex_coord_rect.lower_left().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_right.origin(),
                tex_coord_rect.lower_left().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_right.upper_right(),
                tex_coord_rect.origin().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_right.lower_right(),
                tex_coord_rect.upper_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.upper_right.lower_left(),
                tex_coord_rect.lower_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_left.origin(),
                tex_coord_rect.upper_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_left.upper_right(),
                tex_coord_rect.lower_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_left.lower_right(),
                tex_coord_rect.lower_left().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_left.lower_left(),
                tex_coord_rect.origin().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_right.origin(),
                tex_coord_rect.lower_right().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_right.upper_right(),
                tex_coord_rect.lower_left().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_right.lower_right(),
                tex_coord_rect.origin().to_f32() / corner_size_f,
            ),
            DebugTextureVertex::new(
                corner_rects.lower_right.lower_left(),
                tex_coord_rect.upper_right().to_f32() / corner_size_f,
            ),
        ];

        let mut index_data = Vec::with_capacity(24);
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 0));
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 4));
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 8));
        index_data.extend(QUAD_INDICES.iter().map(|&index| index + 12));

        self.draw_texture_with_vertex_data(
            device,
            allocator,
            &vertex_data,
            &index_data,
            texture,
            color,
        );
    }

    fn corner_texture(&self, filled: bool) -> &Texture {
        if filled {
            &self.corner_fill_texture
        } else {
            &self.corner_outline_texture
        }
    }

    fn draw_texture_with_vertex_data(
        &self,
        device: &Device,
        _allocator: &mut GpuMemoryAllocator,
        vertex_data: &[DebugTextureVertex],
        index_data: &[u32],
        texture: &Texture,
        color: ColorU,
    ) {
        let vertex_buffer = device.create_buffer_with_data(vertex_data, wgpu::BufferUsages::VERTEX);
        let index_buffer = device.create_buffer_with_data(index_data, wgpu::BufferUsages::INDEX);
        {
            let color_uniform = [
                color.r as f32 / 255.0,
                color.g as f32 / 255.0,
                color.b as f32 / 255.0,
                color.a as f32 / 255.0,
            ];

            let color_buffer = device.create_buffer_with_data(
                &color_uniform,
                wgpu::BufferUsages::UNIFORM,
            );

            let bind_group_layout = self.texture_pipeline.get_bind_group_layout(0);
            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("UI Texture BindGroup"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(self.cached_transform_buffer.slice(..).into()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(color_buffer.slice(..).into()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    },
                ],
            });

            if let Some(ref render_target) = self.render_target {
                device.draw_instanced(
                    &RenderTarget::Framebuffer(render_target),
                    &self.texture_pipeline,
                    &[bind_group],
                    &[(&vertex_buffer, 0)],
                    Some((&index_buffer, 0, wgpu::IndexFormat::Uint32)),
                    index_data.len() as u32,
                    1,
                    None,
                );
            }
        }
    }

    pub fn draw_button(
        &mut self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        origin: Vector2I,
        texture: &Texture,
    ) -> bool {
        let button_rect = RectI::new(origin, vec2i(BUTTON_WIDTH, BUTTON_HEIGHT));
        self.draw_solid_rounded_rect(device, allocator, button_rect, WINDOW_COLOR);
        self.draw_rounded_rect_outline(device, allocator, button_rect, OUTLINE_COLOR);
        self.draw_texture(
            device,
            allocator,
            origin + vec2i(PADDING, PADDING),
            texture,
            BUTTON_ICON_COLOR,
        );
        self.event_queue
            .handle_mouse_down_in_rect(button_rect)
            .is_some()
    }

    pub fn draw_text_switch(
        &mut self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        mut origin: Vector2I,
        segment_labels: &[&str],
        mut value: u8,
    ) -> u8 {
        if let Some(new_value) = self.draw_segmented_control(
            device,
            allocator,
            origin,
            Some(value),
            segment_labels.len() as u8,
        ) {
            value = new_value;
        }

        origin = origin + vec2i(0, BUTTON_TEXT_OFFSET);
        for (segment_index, segment_label) in segment_labels.iter().enumerate() {
            let label_width = self.measure_text(segment_label);
            let offset = SEGMENT_SIZE / 2 - label_width / 2;
            self.draw_text(
                device,
                allocator,
                segment_label,
                origin + vec2i(offset, 0),
                segment_index as u8 == value,
            );
            origin += vec2i(SEGMENT_SIZE + 1, 0);
        }

        value
    }

    pub fn draw_image_segmented_control(
        &mut self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        mut origin: Vector2I,
        segment_textures: &[&Texture],
        mut value: Option<u8>,
    ) -> Option<u8> {
        let mut clicked_segment = None;
        if let Some(segment_index) = self.draw_segmented_control(
            device,
            allocator,
            origin,
            value,
            segment_textures.len() as u8,
        ) {
            if let Some(ref mut value) = value {
                *value = segment_index;
            }
            clicked_segment = Some(segment_index);
        }

        for (segment_index, segment_texture) in segment_textures.iter().enumerate() {
            let texture_width = device.texture_size(segment_texture).x();
            let offset = vec2i(SEGMENT_SIZE / 2 - texture_width / 2, PADDING);
            let color = if Some(segment_index as u8) == value {
                WINDOW_COLOR
            } else {
                TEXT_COLOR
            };

            self.draw_texture(device, allocator, origin + offset, segment_texture, color);
            origin += vec2i(SEGMENT_SIZE + 1, 0);
        }

        clicked_segment
    }

    fn draw_segmented_control(
        &mut self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        origin: Vector2I,
        mut value: Option<u8>,
        segment_count: u8,
    ) -> Option<u8> {
        let widget_width = self.measure_segmented_control(segment_count);
        let widget_rect = RectI::new(origin, vec2i(widget_width, BUTTON_HEIGHT));

        let mut clicked_segment = None;
        if let Some(position) = self.event_queue.handle_mouse_down_in_rect(widget_rect) {
            let segment = ((position.x() / (SEGMENT_SIZE + 1)) as u8).min(segment_count - 1);
            if let Some(ref mut value) = value {
                *value = segment;
            }
            clicked_segment = Some(segment);
        }

        self.draw_solid_rounded_rect(device, allocator, widget_rect, WINDOW_COLOR);
        self.draw_rounded_rect_outline(device, allocator, widget_rect, OUTLINE_COLOR);

        if let Some(value) = value {
            let highlight_size = vec2i(SEGMENT_SIZE, BUTTON_HEIGHT);
            let x_offset = value as i32 * SEGMENT_SIZE + (value as i32 - 1);
            self.draw_solid_rounded_rect(
                device,
                allocator,
                RectI::new(origin + vec2i(x_offset, 0), highlight_size),
                TEXT_COLOR,
            );
        }

        let mut segment_origin = origin + vec2i(SEGMENT_SIZE + 1, 0);
        for next_segment_index in 1..segment_count {
            let prev_segment_index = next_segment_index - 1;
            match value {
                Some(value) if value == prev_segment_index || value == next_segment_index => {}
                _ => {
                    self.draw_line(
                        device,
                        allocator,
                        segment_origin,
                        segment_origin + vec2i(0, BUTTON_HEIGHT),
                        TEXT_COLOR,
                    );
                }
            }
            segment_origin += vec2i(SEGMENT_SIZE + 1, 0);
        }

        clicked_segment
    }

    pub fn draw_tooltip(
        &self,
        device: &Device,
        allocator: &mut GpuMemoryAllocator,
        string: &str,
        rect: RectI,
    ) {
        if !rect.to_f32().contains_point(self.mouse_position) {
            return;
        }

        let text_size = self.measure_text(string);
        let window_size = vec2i(text_size + PADDING * 2, TOOLTIP_HEIGHT);
        let origin = rect.origin() - vec2i(0, window_size.y() + PADDING);

        self.draw_solid_rounded_rect(
            device,
            allocator,
            RectI::new(origin, window_size),
            WINDOW_COLOR,
        );
        self.draw_text(
            device,
            allocator,
            string,
            origin + vec2i(PADDING, PADDING + FONT_ASCENT),
            false,
        );
    }
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
#[repr(C)]
struct DebugTextureVertex {
    position_x: f32,
    position_y: f32,
    tex_coord_x: f32,
    tex_coord_y: f32,
}

impl DebugTextureVertex {
    fn new(position: Vector2I, tex_coord: Vector2F) -> DebugTextureVertex {
        DebugTextureVertex {
            position_x: position.x() as f32,
            position_y: position.y() as f32,
            tex_coord_x: tex_coord.x(),
            tex_coord_y: tex_coord.y(),
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
#[repr(C)]
struct DebugSolidVertex {
    position_x: f32,
    position_y: f32,
}

impl DebugSolidVertex {
    fn new(position: Vector2I) -> DebugSolidVertex {
        DebugSolidVertex {
            position_x: position.x() as f32,
            position_y: position.y() as f32,
        }
    }
}

struct CornerRects {
    upper_left: RectI,
    upper_right: RectI,
    lower_left: RectI,
    lower_right: RectI,
}

impl CornerRects {
    fn new(device: &Device, rect: RectI, texture: &Texture) -> CornerRects {
        let size = device.texture_size(texture);
        CornerRects {
            upper_left: RectI::new(rect.origin(), size),
            upper_right: RectI::new(rect.upper_right() - vec2i(size.x(), 0), size),
            lower_left: RectI::new(rect.lower_left() - vec2i(0, size.y()), size),
            lower_right: RectI::new(rect.lower_right() - size, size),
        }
    }
}

fn get_color_uniform(color: ColorU) -> UniformData {
    let color = F32x4::new(
        color.r as f32,
        color.g as f32,
        color.b as f32,
        color.a as f32,
    );
    UniformData::Vec4(color * F32x4::splat(1.0 / 255.0))
}

#[derive(Clone, Copy)]
pub enum UIEvent {
    MouseDown(MousePosition),
    MouseDragged(MousePosition),
    MouseUp,
}

pub struct UIEventQueue {
    events: Vec<UIEvent>,
    dragging: bool,
}

impl UIEventQueue {
    fn new() -> UIEventQueue {
        UIEventQueue { events: vec![], dragging: false }
    }

    pub fn push(&mut self, event: UIEvent) {
        self.events.push(event);
    }

    pub fn drain(&mut self) -> Vec<UIEvent> {
        mem::replace(&mut self.events, vec![])
    }

    pub fn reset_dragging(&mut self) {
        self.dragging = false;
    }

    pub fn handle_mouse_down_in_rect(&mut self, rect: RectI) -> Option<Vector2I> {
        let (mut remaining_events, mut result) = (vec![], None);
        for event in self.events.drain(..) {
            match event {
                UIEvent::MouseDown(position) if rect.contains_point(position.absolute) => {
                    result = Some(position.absolute - rect.origin());
                }
                UIEvent::MouseUp => {
                    self.dragging = false;
                }
                event => remaining_events.push(event),
            }
        }
        self.events = remaining_events;
        result
    }

    pub fn handle_mouse_down_or_dragged_in_rect(&mut self, rect: RectI) -> Option<Vector2I> {
        let (mut remaining_events, mut result) = (vec![], None);
        for event in self.events.drain(..) {
            match event {
                UIEvent::MouseDown(position) => {
                    if rect.contains_point(position.absolute) {
                        self.dragging = true;
                        result = Some(position.absolute - rect.origin());
                    } else {
                        self.dragging = false;
                        remaining_events.push(event);
                    }
                }
                UIEvent::MouseDragged(position) => {
                    if self.dragging {
                        let clamped_x = position.absolute.x().max(rect.origin().x()).min(rect.max_x());
                        let clamped_y = position.absolute.y().max(rect.origin().y()).min(rect.max_y());
                        let clamped_position = Vector2I::new(clamped_x, clamped_y);
                        result = Some(clamped_position - rect.origin());
                    } else if rect.contains_point(position.absolute) {
                        self.dragging = true;
                        result = Some(position.absolute - rect.origin());
                    } else {
                        remaining_events.push(event);
                    }
                }
                UIEvent::MouseUp => {
                    self.dragging = false;
                }
            }
        }
        self.events = remaining_events;
        result
    }
}

#[derive(Clone, Copy)]
pub struct MousePosition {
    pub absolute: Vector2I,
    pub relative: Vector2I,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct DebugFont {
    name: String,
    size: i32,
    bold: bool,
    italic: bool,
    width: u32,
    height: u32,
    characters: HashMap<char, DebugCharacter>,
}

#[derive(Deserialize)]
struct DebugCharacter {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    #[serde(rename = "originX")]
    origin_x: i32,
    #[serde(rename = "originY")]
    origin_y: i32,
    advance: i32,
}

impl DebugFont {
    #[inline]
    fn load(resources: &dyn ResourceLoader) -> DebugFont {
        serde_json::from_slice(&resources.slurp(FONT_JSON_VIRTUAL_PATH).unwrap()).unwrap()
    }
}
