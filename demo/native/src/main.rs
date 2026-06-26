// pathfinder/demo/native/src/main.rs
//
// Copyright © 2026 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A demo app for Pathfinder using winit.

use nfd::Response;
use pathfinder_demo::window::{DataPath, Event, Keycode, SurfaceTextureHandle, View, Window, WindowSize};
use pathfinder_demo::{DemoApp, Options};
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::{vec2i, Vector2I};
use pathfinder_gpu::{Device as PathfinderDevice, Texture};
use pathfinder_resources::embedded::EmbeddedResourceLoader;
use pathfinder_resources::ResourceLoader;
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event as WinitEvent, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window as WinitWindow, WindowBuilder};

const DEFAULT_WINDOW_WIDTH: u32 = 1067;
const DEFAULT_WINDOW_HEIGHT: u32 = 800;

fn main() {
    color_backtrace::install();
    pretty_env_logger::init();

    // Read command line options.
    let mut options = Options::default();
    options.command_line_overrides();

    let event_loop = EventLoop::new().unwrap();
    let window_builder = WindowBuilder::new()
        .with_title("Pathfinder Demo")
        .with_inner_size(LogicalSize::new(DEFAULT_WINDOW_WIDTH as f64, DEFAULT_WINDOW_HEIGHT as f64));

    let window = Arc::new(window_builder.build(&event_loop).unwrap());

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: if options.high_performance_gpu {
            wgpu::PowerPreference::HighPerformance
        } else {
            wgpu::PowerPreference::LowPower
        },
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    })).unwrap();

    // Configure D3D11 backend for native read_write support
    let mut required_features = wgpu::Features::empty();
    if adapter.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
        required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::default(),
        },
    )).unwrap();

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let pathfinder_device = PathfinderDevice::new(device.clone(),
                                                  queue.clone(),
                                                  adapter.get_info().name,
                                                  adapter.get_info().backend.to_str().to_string());

    let mut config = surface.get_default_config(&adapter,
        window.inner_size().width,
        window.inner_size().height).unwrap();
    // Use Rgba8Unorm to match blit pipeline format (instead of default Rgba8UnormSrgb)
    config.format = wgpu::TextureFormat::Rgba8Unorm;
    surface.configure(&device, &config);

    let window_impl = WindowImpl {
        window: window.clone(),
        surface,
        device: device.clone(),
        queue: queue.clone(),
        pathfinder_device,
        config: RefCell::new(config),
        resource_loader: EmbeddedResourceLoader::new(),
        next_user_event_id: Cell::new(0),
        pending_open_path: RefCell::new(None),
    };

    let window_size = window_impl.size();
    let mut app = DemoApp::new(window_impl, window_size, options);

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut last_mouse_position = vec2i(0, 0);
    let mut mouse_pressed = false;
    let mut pending_events = vec![];

    event_loop.run(move |event, window_target| {
        match event {
            WinitEvent::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => window_target.exit(),
                WindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        let mut config = app.window.config.borrow_mut();
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        config.format = wgpu::TextureFormat::Rgba8Unorm;
                        app.window.surface.configure(&app.window.device, &config);
                        pending_events.push(Event::WindowResized(app.window.size()));
                        app.window.window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    {
                        let config = app.window.config.borrow();
                        if config.width == 0 || config.height == 0 {
                            return;
                        }
                    }
                    if let Some(path) = app.window.pending_open_path.borrow_mut().take() {
                        pending_events.push(Event::OpenData(DataPath::Path(path)));
                    }
                    let scene_count = app.prepare_frame(pending_events.drain(..).collect());
                    app.draw_scene();
                    app.begin_compositing();
                    for scene_index in 0..scene_count {
                        app.composite_scene(scene_index);
                    }
                    app.finish_drawing_frame();
                }
                _ => {
                    if let Some(pf_event) = map_winit_event(&event, &mut last_mouse_position, &mut mouse_pressed) {
                        pending_events.push(pf_event);
                        app.window.window.request_redraw();
                    }
                }
            }
            WinitEvent::AboutToWait => {
                app.window.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}

struct WindowImpl {
    window: Arc<WinitWindow>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pathfinder_device: PathfinderDevice,
    config: RefCell<wgpu::SurfaceConfiguration>,
    resource_loader: EmbeddedResourceLoader,
    next_user_event_id: Cell<u32>,
    pending_open_path: RefCell<Option<PathBuf>>,
}

impl Window for WindowImpl {
    fn device(&self) -> &PathfinderDevice {
        &self.pathfinder_device
    }

    fn present(&mut self) {
        // Deprecated - use present_texture instead
    }

    fn present_texture(&mut self, texture: &Texture) {
        // Deprecated - blit is now handled by renderer.blit_to_surface()
    }

    fn get_current_surface(&mut self) -> Option<SurfaceTextureHandle> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(st) => st,
            wgpu::CurrentSurfaceTexture::Suboptimal(st) => st,
            wgpu::CurrentSurfaceTexture::Timeout => return None,
            wgpu::CurrentSurfaceTexture::Outdated => {
                let config = self.config.borrow().clone();
                self.surface.configure(self.device.as_ref(), &config);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return None;
            }
            wgpu::CurrentSurfaceTexture::Occluded => return None,
            wgpu::CurrentSurfaceTexture::Validation => return None,
        };

        let config = self.config.borrow();
        let size = vec2i(config.width as i32, config.height as i32);
        Some(SurfaceTextureHandle::new(surface_texture, size))
    }

    fn make_current(&mut self, _view: View) {}

    fn viewport(&self, view: View) -> RectI {
        let config = self.config.borrow();
        let size = vec2i(config.width as i32, config.height as i32);
        let mut x_offset = 0;
        let mut viewport_size = size;
        if let View::Stereo(index) = view {
            viewport_size.set_x(size.x() / 2);
            x_offset = viewport_size.x() * (index as i32);
        }
        RectI::new(vec2i(x_offset, 0), viewport_size)
    }

    fn resource_loader(&self) -> &dyn ResourceLoader {
        &self.resource_loader
    }

    fn create_user_event_id(&self) -> u32 {
        let id = self.next_user_event_id.get();
        self.next_user_event_id.set(id + 1);
        id
    }

    fn push_user_event(_message_type: u32, _message_data: u32) {
        // TODO: proxy.send_event
    }

    fn present_open_svg_dialog(&mut self) {
        if let Ok(Response::Okay(path)) = nfd::open_file_dialog(Some("svg"), None) {
            *self.pending_open_path.borrow_mut() = Some(PathBuf::from(path));
            self.window.request_redraw();
        }
    }

    fn run_save_dialog(&self, extension: &str) -> Result<PathBuf, ()> {
        match nfd::open_save_dialog(Some(extension), None) {
            Ok(Response::Okay(path)) => Ok(PathBuf::from(path)),
            _ => Err(()),
        }
    }
}

impl WindowImpl {
    fn size(&self) -> WindowSize {
        WindowSize {
            physical_size: vec2i(self.window.inner_size().width as i32, self.window.inner_size().height as i32),
            scale_factor: self.window.scale_factor() as f32,
        }
    }
}

fn map_winit_event(event: &WindowEvent, last_mouse_position: &mut Vector2I, mouse_pressed: &mut bool) -> Option<Event> {
    match event {
        WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state, .. }, .. } => {
            let keycode = match logical_key {
                Key::Named(NamedKey::Escape) => Keycode::Escape,
                Key::Named(NamedKey::Tab) => Keycode::Tab,
                Key::Character(c) => Keycode::Alphanumeric(c.as_bytes()[0]),
                _ => return None,
            };
            match state {
                ElementState::Pressed => Some(Event::KeyDown(keycode)),
                ElementState::Released => Some(Event::KeyUp(keycode)),
            }
        }
        WindowEvent::CursorMoved { position, .. } => {
            let new_position = vec2i(position.x as i32, position.y as i32);
            *last_mouse_position = new_position;
            if *mouse_pressed {
                Some(Event::MouseDragged(new_position))
            } else {
                Some(Event::MouseMoved(new_position))
            }
        }
        WindowEvent::MouseInput { state, button, .. } => {
            if *button == MouseButton::Left {
                match state {
                    ElementState::Pressed => {
                        *mouse_pressed = true;
                        Some(Event::MouseDown(*last_mouse_position))
                    }
                    ElementState::Released => {
                        *mouse_pressed = false;
                        Some(Event::MouseUp)
                    }
                }
            } else {
                None
            }
        }
        // WindowEvent::MouseWheel { delta, .. } => {
        //     let d_dist = match delta {
        //         winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
        //         winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.05,
        //     };
        //     Some(Event::Zoom(d_dist, *last_mouse_position))
        // }
        _ => None,
    }
}
