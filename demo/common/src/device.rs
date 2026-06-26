// pathfinder/demo/common/src/device.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! GPU rendering code specifically for the demo.

use pathfinder_gpu::Device;
use pathfinder_resources::ResourceLoader;

pub struct GroundProgram {
    pub pipeline: wgpu::RenderPipeline,
    pub transform_uniform: u32,
    pub gridline_count_uniform: u32,
    pub ground_color_uniform: u32,
    pub gridline_color_uniform: u32,
}

impl GroundProgram {
    pub fn new(device: &Device, resources: &dyn ResourceLoader) -> GroundProgram {
        let pipeline = device.create_render_pipeline(resources, "demo_ground", None);
        GroundProgram {
            pipeline,
            transform_uniform: 0,
            gridline_count_uniform: 1,
            ground_color_uniform: 2,
            gridline_color_uniform: 3,
        }
    }
}

pub struct GroundVertexArray {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

impl GroundVertexArray {
    pub fn new(
        _device: &Device,
        _ground_program: &GroundProgram,
        quad_vertex_positions_buffer: &wgpu::Buffer,
        quad_vertex_indices_buffer: &wgpu::Buffer,
    ) -> GroundVertexArray {
        GroundVertexArray {
            vertex_buffer: quad_vertex_positions_buffer.clone(),
            index_buffer: quad_vertex_indices_buffer.clone(),
        }
    }
}
