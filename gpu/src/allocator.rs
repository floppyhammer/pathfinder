// pathfinder/gpu/src/gpu/allocator.rs
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! GPU memory management.

use crate::{Device, Texture};
use fxhash::FxHashMap;
use instant::Instant;
use pathfinder_geometry::vector::Vector2I;
use std::collections::VecDeque;
use std::mem;

// Everything above 16 MB is allocated exactly.
const MAX_BUFFER_SIZE_CLASS: u64 = 16 * 1024 * 1024;

// Number of seconds before unused memory is purged.
//
// TODO(pcwalton): jemalloc uses a sigmoidal decay curve here. Consider something similar.
const DECAY_TIME: f32 = 0.250;

// Number of frames to wait before we can reuse an object.
//
// This helps avoid stalls and ensure that the GPU is no longer using the resource.
const MAX_FRAMES_IN_FLIGHT: u64 = 3;

pub struct GpuMemoryAllocator {
    general_buffers_in_use: FxHashMap<GeneralBufferID, BufferAllocation>,
    index_buffers_in_use: FxHashMap<IndexBufferID, BufferAllocation>,
    textures_in_use: FxHashMap<TextureID, TextureAllocation>,
    free_objects: VecDeque<FreeObject>,
    next_general_buffer_id: GeneralBufferID,
    next_index_buffer_id: IndexBufferID,
    next_texture_id: TextureID,
    bytes_committed: u64,
    bytes_allocated: u64,
    current_frame: u64,
}

struct BufferAllocation {
    buffer: wgpu::Buffer,
    size: u64,
    tag: BufferTag,
}

struct TextureAllocation {
    texture: Texture,
    descriptor: TextureDescriptor,
    tag: TextureTag,
}

struct FreeObject {
    timestamp: Instant,
    frame: u64,
    kind: FreeObjectKind,
}

enum FreeObjectKind {
    GeneralBuffer {
        id: GeneralBufferID,
        allocation: BufferAllocation,
    },
    IndexBuffer {
        id: IndexBufferID,
        allocation: BufferAllocation,
    },
    Texture {
        id: TextureID,
        allocation: TextureAllocation,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

// Vertex or storage buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeneralBufferID(pub u64);

// Index buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexBufferID(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureID(pub u64);

// For debugging and profiling.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BufferTag(pub &'static str);

// For debugging and profiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTag(pub &'static str);

impl GpuMemoryAllocator {
    pub fn new() -> GpuMemoryAllocator {
        GpuMemoryAllocator {
            general_buffers_in_use: FxHashMap::default(),
            index_buffers_in_use: FxHashMap::default(),
            textures_in_use: FxHashMap::default(),
            free_objects: VecDeque::new(),
            next_general_buffer_id: GeneralBufferID(0),
            next_index_buffer_id: IndexBufferID(0),
            next_texture_id: TextureID(0),
            bytes_committed: 0,
            bytes_allocated: 0,
            current_frame: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
    }

    pub fn allocate_general_buffer<T>(
        &mut self,
        device: &Device,
        size: u64,
        tag: BufferTag,
    ) -> GeneralBufferID {
        let mut byte_size = size * size_of::<T>() as u64;
        if byte_size < MAX_BUFFER_SIZE_CLASS {
            byte_size = byte_size.next_power_of_two();
        }

        for free_object_index in 0..self.free_objects.len() {
            match self.free_objects[free_object_index] {
                FreeObject {
                    frame,
                    kind: FreeObjectKind::GeneralBuffer { ref allocation, .. },
                    ..
                } if allocation.size == byte_size
                    && self.current_frame - frame >= MAX_FRAMES_IN_FLIGHT => {}
                _ => continue,
            }

            let (id, mut allocation) = match self.free_objects.remove(free_object_index) {
                Some(FreeObject {
                    kind: FreeObjectKind::GeneralBuffer { id, allocation },
                    ..
                }) => (id, allocation),
                _ => unreachable!(),
            };

            allocation.tag = tag;
            self.bytes_committed += allocation.size;
            self.general_buffers_in_use.insert(id, allocation);
            return id;
        }

        let buffer = device.create_buffer(
            byte_size,
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let id = self.next_general_buffer_id;
        self.next_general_buffer_id.0 += 1;

        debug!(
            "mapping general buffer: {:?} {} ({}x{}) {:?}",
            id,
            byte_size,
            size,
            size_of::<T>(),
            tag
        );

        self.general_buffers_in_use.insert(
            id,
            BufferAllocation {
                buffer,
                size: byte_size,
                tag,
            },
        );
        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub fn allocate_index_buffer<T>(
        &mut self,
        device: &Device,
        size: u64,
        tag: BufferTag,
    ) -> IndexBufferID {
        let mut byte_size = size * mem::size_of::<T>() as u64;
        if byte_size < MAX_BUFFER_SIZE_CLASS {
            byte_size = byte_size.next_power_of_two();
        }

        for free_object_index in 0..self.free_objects.len() {
            match self.free_objects[free_object_index] {
                FreeObject {
                    frame,
                    kind: FreeObjectKind::IndexBuffer { ref allocation, .. },
                    ..
                } if allocation.size == byte_size
                    && self.current_frame - frame >= MAX_FRAMES_IN_FLIGHT => {}
                _ => continue,
            }

            let (id, mut allocation) = match self.free_objects.remove(free_object_index) {
                Some(FreeObject {
                    kind: FreeObjectKind::IndexBuffer { id, allocation },
                    ..
                }) => (id, allocation),
                _ => unreachable!(),
            };

            allocation.tag = tag;
            self.bytes_committed += allocation.size;
            self.index_buffers_in_use.insert(id, allocation);
            return id;
        }

        let buffer = device.create_buffer(
            byte_size,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        );

        let id = self.next_index_buffer_id;
        self.next_index_buffer_id.0 += 1;

        debug!(
            "mapping index buffer: {:?} {} ({}x{}) {:?}",
            id,
            byte_size,
            size,
            mem::size_of::<T>(),
            tag
        );

        self.index_buffers_in_use.insert(
            id,
            BufferAllocation {
                buffer,
                size: byte_size,
                tag,
            },
        );
        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub fn allocate_texture(
        &mut self,
        device: &Device,
        size: Vector2I,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        tag: TextureTag,
    ) -> TextureID {
        let descriptor = TextureDescriptor {
            width: size.x() as u32,
            height: size.y() as u32,
            format,
            usage,
        };
        let byte_size = descriptor.byte_size();

        for free_object_index in 0..self.free_objects.len() {
            match self.free_objects[free_object_index] {
                FreeObject {
                    frame,
                    kind: FreeObjectKind::Texture { ref allocation, .. },
                    ..
                } if allocation.descriptor == descriptor
                    && self.current_frame - frame >= MAX_FRAMES_IN_FLIGHT => {}
                _ => continue,
            }

            let (id, mut allocation) = match self.free_objects.remove(free_object_index) {
                Some(FreeObject {
                    kind: FreeObjectKind::Texture { id, allocation },
                    ..
                }) => (id, allocation),
                _ => unreachable!(),
            };

            allocation.tag = tag;
            self.bytes_committed += allocation.descriptor.byte_size();
            self.textures_in_use.insert(id, allocation);
            return id;
        }

        debug!("mapping texture: {:?} {:?}", descriptor, tag);

        let texture = device.create_texture(format, size, descriptor.usage);
        let id = self.next_texture_id;
        self.next_texture_id.0 += 1;

        self.textures_in_use.insert(
            id,
            TextureAllocation {
                texture,
                descriptor,
                tag,
            },
        );

        self.bytes_allocated += byte_size;
        self.bytes_committed += byte_size;

        id
    }

    pub fn purge_if_needed(&mut self) {
        let now = Instant::now();
        loop {
            match self.free_objects.front() {
                Some(FreeObject { timestamp, .. })
                    if (now - *timestamp).as_secs_f32() >= DECAY_TIME => {}
                _ => break,
            }
            match self.free_objects.pop_front() {
                None => break,
                Some(FreeObject {
                    kind: FreeObjectKind::GeneralBuffer { allocation, .. },
                    ..
                }) => {
                    debug!("purging general buffer: {}", allocation.size);
                    self.bytes_allocated -= allocation.size;
                }
                Some(FreeObject {
                    kind: FreeObjectKind::IndexBuffer { allocation, .. },
                    ..
                }) => {
                    debug!("purging index buffer: {}", allocation.size);
                    self.bytes_allocated -= allocation.size;
                }
                Some(FreeObject {
                    kind: FreeObjectKind::Texture { allocation, .. },
                    ..
                }) => {
                    debug!("purging texture: {:?}", allocation.descriptor);
                    self.bytes_allocated -= allocation.descriptor.byte_size();
                }
            }
        }
    }

    pub fn free_general_buffer(&mut self, id: GeneralBufferID) {
        let allocation = self
            .general_buffers_in_use
            .remove(&id)
            .expect("Attempted to free unallocated general buffer!");
        self.bytes_committed -= allocation.size;
        self.free_objects.push_back(FreeObject {
            timestamp: Instant::now(),
            frame: self.current_frame,
            kind: FreeObjectKind::GeneralBuffer { id, allocation },
        });
    }

    pub fn free_index_buffer(&mut self, id: IndexBufferID) {
        let allocation = self
            .index_buffers_in_use
            .remove(&id)
            .expect("Attempted to free unallocated index buffer!");
        self.bytes_committed -= allocation.size;
        self.free_objects.push_back(FreeObject {
            timestamp: Instant::now(),
            frame: self.current_frame,
            kind: FreeObjectKind::IndexBuffer { id, allocation },
        });
    }

    pub fn free_texture(&mut self, id: TextureID) {
        let allocation = self
            .textures_in_use
            .remove(&id)
            .expect("Attempted to free unallocated texture!");
        let byte_size = allocation.descriptor.byte_size();
        self.bytes_committed -= byte_size;
        self.free_objects.push_back(FreeObject {
            timestamp: Instant::now(),
            frame: self.current_frame,
            kind: FreeObjectKind::Texture { id, allocation },
        });
    }

    pub fn get_general_buffer(&self, id: GeneralBufferID) -> &wgpu::Buffer {
        &self.general_buffers_in_use[&id].buffer
    }

    pub fn get_index_buffer(&self, id: IndexBufferID) -> &wgpu::Buffer {
        &self.index_buffers_in_use[&id].buffer
    }

    pub fn get_texture(&self, id: TextureID) -> &Texture {
        &self.textures_in_use[&id].texture
    }

    #[inline]
    pub fn bytes_allocated(&self) -> u64 {
        self.bytes_allocated
    }

    #[inline]
    pub fn bytes_committed(&self) -> u64 {
        self.bytes_committed
    }

    #[allow(dead_code)]
    pub fn dump(&self) {
        println!("GPU memory dump");
        println!("---------------");

        println!("General buffers:");
        let mut ids: Vec<GeneralBufferID> = self.general_buffers_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.general_buffers_in_use[&id];
            println!(
                "id {:?}: {:?} ({:?} B)",
                id, allocation.tag, allocation.size
            );
        }

        println!("Index buffers:");
        let mut ids: Vec<IndexBufferID> = self.index_buffers_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.index_buffers_in_use[&id];
            println!(
                "id {:?}: {:?} ({:?} B)",
                id, allocation.tag, allocation.size
            );
        }

        println!("Textures:");
        let mut ids: Vec<TextureID> = self.textures_in_use.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let allocation = &self.textures_in_use[&id];
            println!(
                "id {:?}: {:?} {:?}x{:?} {:?} ({:?} B)",
                id,
                allocation.tag,
                allocation.descriptor.width,
                allocation.descriptor.height,
                allocation.descriptor.format,
                allocation.descriptor.byte_size()
            );
        }
    }
}

impl TextureDescriptor {
    fn byte_size(&self) -> u64 {
        let block_size = match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::R16Float => 2,
            wgpu::TextureFormat::Rgba8Unorm => 4,
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rgba32Float => 16,
            _ => 4, // Default fallback
        };
        self.width as u64 * self.height as u64 * block_size
    }
}
