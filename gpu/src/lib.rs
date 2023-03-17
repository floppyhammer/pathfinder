// pathfinder/gpu/src/lib.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Minimal abstractions over GPU device capabilities.

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;

use image::{DynamicImage, GenericImageView, ImageFormat};
use pathfinder_geometry::vector::vec2i;
use pathfinder_resources::ResourceLoader;

pub mod allocator;

#[derive(Clone, Copy, Debug)]
pub enum RenderTarget<'a> {
    Default,
    Framebuffer(&'a wgpu::TextureView),
}

fn create_texture_from_png(resources: &dyn ResourceLoader,
                           name: &str,
                           format: wgpu::TextureFormat)
                           -> Self::Texture {
    let data = resources.slurp(&format!("textures/{}.png", name)).unwrap();
    let image = image::load_from_memory_with_format(&data, ImageFormat::Png).unwrap();
    match format {
        wgpu::TextureFormat::R8Unorm => {
            let image = image.to_luma8();
            let size = vec2i(image.width() as i32, image.height() as i32);

            self.create_texture_from_data(format, size, TextureDataRef::U8(&image))
        }
        wgpu::TextureFormat::Rgba8Unorm => {
            let image = image.to_rgba8();
            let size = vec2i(image.width() as i32, image.height() as i32);
            self.create_texture_from_data(format, size, TextureDataRef::U8(&image))
        }
        _ => unimplemented!(),
    }
}

pub fn upload_png_to_texture(queue: &wgpu::Device,
                             resources: &dyn ResourceLoader,
                             name: &str,
                             texture: &wgpu::Texture,
                             format: wgpu::TextureFormat) {
    let data = resources.slurp(&format!("textures/{}.png", name)).unwrap();
    let image = image::load_from_memory_with_format(&data, ImageFormat::Png).unwrap();

    let data: &[u8];
    let bytes_per_row;

    match image {
        DynamicImage::ImageLuma8(gray) => {
            data = &gray;
            bytes_per_row = image.width();
        }
        DynamicImage::ImageRgba8(rgba) => {
            data = &rgba;
            bytes_per_row = image.width() * 4;
        }
        _ => {
            unimplemented!()
        }
    }

    let img_copy_texture = wgpu::ImageCopyTexture {
        aspect: wgpu::TextureAspect::All,
        texture: &texture,
        mip_level: 0,
        origin: wgpu::Origin3d {
            x: region.min_x() as u32,
            y: region.min_y() as u32,
            z: 0,
        },
    };

    let size = wgpu::Extent3d {
        width: image.width(),
        height: image.height(),
        depth_or_array_layers: 1,
    };

    queue.write_texture(
        img_copy_texture,
        data,
        wgpu::ImageDataLayout {
            offset: 0 as wgpu::BufferAddress,
            bytes_per_row: std::num::NonZeroU32::new(bytes_per_row),
            rows_per_image: std::num::NonZeroU32::new(image.height()),
        },
        size,
    );
}

fn create_program_from_shader_names(
    resources: &dyn ResourceLoader,
    program_name: &str,
    shader_names: ProgramKind<&str>,
) -> Self::Program {
    let shaders = match shader_names {
        ProgramKind::Raster { vertex, fragment } => {
            ProgramKind::Raster {
                vertex: self.create_shader(resources, vertex, ShaderKind::Vertex),
                fragment: self.create_shader(resources, fragment, ShaderKind::Fragment),
            }
        }
        ProgramKind::Compute(compute) => {
            ProgramKind::Compute(self.create_shader(resources, compute, ShaderKind::Compute))
        }
    };
    self.create_program_from_shaders(resources, program_name, shaders)
}

fn create_raster_program(&self, resources: &dyn ResourceLoader, name: &str) -> Self::Program {
    let shaders = ProgramKind::Raster { vertex: name, fragment: name };
    self.create_program_from_shader_names(resources, name, shaders)
}

fn create_compute_program(&self, resources: &dyn ResourceLoader, name: &str) -> Self::Program {
    let shaders = ProgramKind::Compute(name);
    self.create_program_from_shader_names(resources, name, shaders)
}
