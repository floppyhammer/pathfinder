// pathfinder/path-utils/src/lib.rs
//
// Copyright © 2017 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate euclid;
extern crate freetype_sys;
#[macro_use]
extern crate serde_derive;

use euclid::{Point2D, Transform2D};
use std::u32;

pub mod freetype;
pub mod stroke;

#[derive(Clone, Copy, Debug)]
pub enum PathSegment {
    MoveTo(Point2D<f32>),
    LineTo(Point2D<f32>),
    CurveTo(Point2D<f32>, Point2D<f32>),
    ClosePath,
}

#[derive(Clone, Debug)]
pub struct PathBuffer {
    pub endpoints: Vec<Endpoint>,
    pub control_points: Vec<Point2D<f32>>,
    pub subpaths: Vec<Subpath>,
}

impl PathBuffer {
    #[inline]
    pub fn new() -> PathBuffer {
        PathBuffer {
            endpoints: vec![],
            control_points: vec![],
            subpaths: vec![],
        }
    }

    pub fn add_stream<I>(&mut self, stream: I) where I: Iterator<Item = PathSegment> {
        let mut first_subpath_endpoint_index = self.endpoints.len() as u32;
        for segment in stream {
            match segment {
                PathSegment::ClosePath => self.close_subpath(&mut first_subpath_endpoint_index),

                PathSegment::MoveTo(position) => {
                    self.close_subpath(&mut first_subpath_endpoint_index);
                    self.endpoints.push(Endpoint {
                        position: position,
                        control_point_index: u32::MAX,
                        subpath_index: self.subpaths.len() as u32,
                    })
                }

                PathSegment::LineTo(position) => {
                    self.endpoints.push(Endpoint {
                        position: position,
                        control_point_index: u32::MAX,
                        subpath_index: self.subpaths.len() as u32,
                    })
                }

                PathSegment::CurveTo(control_point_position, endpoint_position) => {
                    let control_point_index = self.control_points.len() as u32;
                    self.control_points.push(control_point_position);
                    self.endpoints.push(Endpoint {
                        position: endpoint_position,
                        control_point_index: control_point_index,
                        subpath_index: self.subpaths.len() as u32,
                    })
                }
            }
        }

        self.close_subpath(&mut first_subpath_endpoint_index)
    }

    fn close_subpath(&mut self, first_subpath_endpoint_index: &mut u32) {
        let last_subpath_endpoint_index = self.endpoints.len() as u32;
        if *first_subpath_endpoint_index != last_subpath_endpoint_index {
            self.subpaths.push(Subpath {
                first_endpoint_index: *first_subpath_endpoint_index,
                last_endpoint_index: last_subpath_endpoint_index,
            })
        }

        *first_subpath_endpoint_index = last_subpath_endpoint_index;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Endpoint {
    pub position: Point2D<f32>,
    /// `u32::MAX` if not present.
    pub control_point_index: u32,
    pub subpath_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Subpath {
    pub first_endpoint_index: u32,
    pub last_endpoint_index: u32,
}

pub struct Transform2DPathStream<I> {
    inner: I,
    transform: Transform2D<f32>,
}

impl<I> Transform2DPathStream<I> where I: Iterator<Item = PathSegment> {
    pub fn new(inner: I, transform: &Transform2D<f32>) -> Transform2DPathStream<I> {
        Transform2DPathStream {
            inner: inner,
            transform: *transform,
        }
    }
}

impl<I> Iterator for Transform2DPathStream<I> where I: Iterator<Item = PathSegment> {
    type Item = PathSegment;

    fn next(&mut self) -> Option<PathSegment> {
        match self.inner.next() {
            None => None,
            Some(PathSegment::MoveTo(position)) => {
                Some(PathSegment::MoveTo(self.transform.transform_point(&position)))
            }
            Some(PathSegment::LineTo(position)) => {
                Some(PathSegment::LineTo(self.transform.transform_point(&position)))
            }
            Some(PathSegment::CurveTo(control_point_position, endpoint_position)) => {
                Some(PathSegment::CurveTo(self.transform.transform_point(&control_point_position),
                                          self.transform.transform_point(&endpoint_position)))
            }
            Some(PathSegment::ClosePath) => Some(PathSegment::ClosePath),
        }
    }
}
