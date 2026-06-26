// pathfinder/demo/common/src/window.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A minimal cross-platform windowing layer.

use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::transform3d::{Perspective, Transform4F};
use pathfinder_geometry::vector::Vector2I;
use pathfinder_gpu::{Device, Texture};
use pathfinder_resources::ResourceLoader;
use rayon::ThreadPoolBuilder;
use std::ops::Deref;
use std::path::PathBuf;
use wgpu;

pub trait Window {
    fn device(&self) -> &Device;
    fn present(&mut self);
    /// Present the given texture to the screen surface.
    /// This method should blit the texture to the swapchain and call surface.present().
    fn present_texture(&mut self, texture: &Texture);

    /// Get the current surface texture for rendering.
    /// Returns the surface texture, its view, and the size.
    fn get_current_surface(&mut self) -> Option<SurfaceTextureHandle>;

    fn make_current(&mut self, view: View);
    fn viewport(&self, view: View) -> RectI;
    fn resource_loader(&self) -> &dyn ResourceLoader;
    fn create_user_event_id(&self) -> u32;
    fn push_user_event(message_type: u32, message_data: u32);
    fn present_open_svg_dialog(&mut self);
    fn run_save_dialog(&self, extension: &str) -> Result<PathBuf, ()>;

    fn adjust_thread_pool_settings(&self, builder: ThreadPoolBuilder) -> ThreadPoolBuilder {
        builder
    }
}

/// Handle for the current surface texture.
/// Implementors should ensure the texture is presented when dropped.
pub struct SurfaceTextureHandle {
    surface_texture: Option<wgpu::SurfaceTexture>,
    view: wgpu::TextureView,
    size: Vector2I,
}

impl SurfaceTextureHandle {
    pub fn new(surface_texture: wgpu::SurfaceTexture, size: Vector2I) -> Self {
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        SurfaceTextureHandle {
            surface_texture: Some(surface_texture),
            view,
            size,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn size(&self) -> Vector2I {
        self.size
    }
}

impl Deref for SurfaceTextureHandle {
    type Target = wgpu::SurfaceTexture;

    fn deref(&self) -> &Self::Target {
        self.surface_texture.as_ref().unwrap()
    }
}

impl Drop for SurfaceTextureHandle {
    fn drop(&mut self) {
        if let Some(st) = self.surface_texture.take() {
            st.present();
        }
    }
}

pub enum Event {
    Quit,
    WindowResized(WindowSize),
    KeyDown(Keycode),
    KeyUp(Keycode),
    MouseDown(Vector2I),
    MouseUp,
    MouseMoved(Vector2I),
    MouseDragged(Vector2I),
    Zoom(f32, Vector2I),
    Look {
        pitch: f32,
        yaw: f32,
    },
    SetEyeTransforms(Vec<OcularTransform>),
    OpenData(DataPath),
    User {
        message_type: u32,
        message_data: u32,
    },
}

#[derive(Clone, Copy)]
pub enum Keycode {
    Alphanumeric(u8),
    Escape,
    Tab,
}

#[derive(Clone, Copy, Debug)]
pub struct WindowSize {
    pub physical_size: Vector2I,
    pub scale_factor: f32,
}

impl WindowSize {
    #[inline]
    pub fn device_size(&self) -> Vector2I {
        self.physical_size.to_f32().to_i32()
    }

    #[inline]
    pub fn logical_size(&self) -> Vector2I {
        (self.physical_size.to_f32() / self.scale_factor).to_i32()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum View {
    Mono,
    Stereo(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct OcularTransform {
    // The perspective which converts from camera coordinates to display coordinates
    pub perspective: Perspective,

    // The view transform which converts from world coordinates to camera coordinates
    pub modelview_to_eye: Transform4F,
}

#[derive(Clone)]
pub enum DataPath {
    Default,
    Resource(String),
    Path(PathBuf),
}
