// pathfinder/renderer/src/gpu/blend.rs
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helpers for blending.

use wgpu::{BlendComponent, BlendFactor, BlendState};
use crate::gpu_data::ColorCombineMode;
use crate::paint::PaintCompositeOp;
use pathfinder_content::effects::BlendMode;
use pathfinder_gpu::{BlendFactor, BlendState};

const COMBINER_CTRL_COLOR_COMBINE_SRC_IN:  i32 = 0x1;
const COMBINER_CTRL_COLOR_COMBINE_DEST_IN: i32 = 0x2;

const COMBINER_CTRL_COMPOSITE_NORMAL: i32 =         0x0;
const COMBINER_CTRL_COMPOSITE_MULTIPLY: i32 =       0x1;
const COMBINER_CTRL_COMPOSITE_SCREEN: i32 =         0x2;
const COMBINER_CTRL_COMPOSITE_OVERLAY: i32 =        0x3;
const COMBINER_CTRL_COMPOSITE_DARKEN: i32 =         0x4;
const COMBINER_CTRL_COMPOSITE_LIGHTEN: i32 =        0x5;
const COMBINER_CTRL_COMPOSITE_COLOR_DODGE: i32 =    0x6;
const COMBINER_CTRL_COMPOSITE_COLOR_BURN: i32 =     0x7;
const COMBINER_CTRL_COMPOSITE_HARD_LIGHT: i32 =     0x8;
const COMBINER_CTRL_COMPOSITE_SOFT_LIGHT: i32 =     0x9;
const COMBINER_CTRL_COMPOSITE_DIFFERENCE: i32 =     0xa;
const COMBINER_CTRL_COMPOSITE_EXCLUSION: i32 =      0xb;
const COMBINER_CTRL_COMPOSITE_HUE: i32 =            0xc;
const COMBINER_CTRL_COMPOSITE_SATURATION: i32 =     0xd;
const COMBINER_CTRL_COMPOSITE_COLOR: i32 =          0xe;
const COMBINER_CTRL_COMPOSITE_LUMINOSITY: i32 =     0xf;

pub(crate) trait ToBlendState {
    fn to_blend_state(self) -> Option<BlendState>;
}

impl ToBlendState for BlendMode {
    fn to_blend_state(self) -> Option<BlendState> {
        match self {
            BlendMode::Clear => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::SrcOver => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::DestOver => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusDestAlpha,
                        dst_factor: BlendFactor::One,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::OneMinusDestAlpha,
                        dst_factor: BlendFactor::One,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::SrcIn => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::DstAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::DstAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::DestIn => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::SrcAlpha,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::SrcAlpha,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::SrcOut => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::DestOut => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusSrcAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::OneMinusSrcAlpha,
                        dst_factor: BlendFactor::Zero,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::SrcAtop => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::DstAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::DstAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::DestAtop => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::SrcAlpha,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::SrcAlpha,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::Xor => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::Lighter => {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: Default::default(),
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: Default::default(),
                    },
                })
            }
            BlendMode::Copy |
            BlendMode::Darken |
            BlendMode::Lighten |
            BlendMode::Multiply |
            BlendMode::Screen |
            BlendMode::HardLight |
            BlendMode::Overlay |
            BlendMode::ColorDodge |
            BlendMode::ColorBurn |
            BlendMode::SoftLight |
            BlendMode::Difference |
            BlendMode::Exclusion |
            BlendMode::Hue |
            BlendMode::Saturation |
            BlendMode::Color |
            BlendMode::Luminosity => {
                // Blending is done manually in the shader.
                None
            }
        }
    }
}

pub(crate) trait ToCompositeCtrl {
    fn to_composite_ctrl(&self) -> i32;
}

impl ToCompositeCtrl for BlendMode {
    fn to_composite_ctrl(&self) -> i32 {
        match *self {
            BlendMode::SrcOver |
            BlendMode::SrcAtop |
            BlendMode::DestOver |
            BlendMode::DestOut |
            BlendMode::Xor |
            BlendMode::Lighter |
            BlendMode::Clear |
            BlendMode::Copy |
            BlendMode::SrcIn |
            BlendMode::SrcOut |
            BlendMode::DestIn |
            BlendMode::DestAtop => COMBINER_CTRL_COMPOSITE_NORMAL,
            BlendMode::Multiply => COMBINER_CTRL_COMPOSITE_MULTIPLY,
            BlendMode::Darken => COMBINER_CTRL_COMPOSITE_DARKEN,
            BlendMode::Lighten => COMBINER_CTRL_COMPOSITE_LIGHTEN,
            BlendMode::Screen => COMBINER_CTRL_COMPOSITE_SCREEN,
            BlendMode::Overlay => COMBINER_CTRL_COMPOSITE_OVERLAY,
            BlendMode::ColorDodge => COMBINER_CTRL_COMPOSITE_COLOR_DODGE,
            BlendMode::ColorBurn => COMBINER_CTRL_COMPOSITE_COLOR_BURN,
            BlendMode::HardLight => COMBINER_CTRL_COMPOSITE_HARD_LIGHT,
            BlendMode::SoftLight => COMBINER_CTRL_COMPOSITE_SOFT_LIGHT,
            BlendMode::Difference => COMBINER_CTRL_COMPOSITE_DIFFERENCE,
            BlendMode::Exclusion => COMBINER_CTRL_COMPOSITE_EXCLUSION,
            BlendMode::Hue => COMBINER_CTRL_COMPOSITE_HUE,
            BlendMode::Saturation => COMBINER_CTRL_COMPOSITE_SATURATION,
            BlendMode::Color => COMBINER_CTRL_COMPOSITE_COLOR,
            BlendMode::Luminosity => COMBINER_CTRL_COMPOSITE_LUMINOSITY,
        }
    }
}

impl ToCompositeCtrl for ColorCombineMode {
    fn to_composite_ctrl(&self) -> i32 {
        match *self {
            ColorCombineMode::None => 0,
            ColorCombineMode::SrcIn => COMBINER_CTRL_COLOR_COMBINE_SRC_IN,
            ColorCombineMode::DestIn => COMBINER_CTRL_COLOR_COMBINE_DEST_IN,
        }
    }
}

pub trait BlendModeExt {
    fn needs_readable_framebuffer(self) -> bool;
}

impl BlendModeExt for BlendMode {
    fn needs_readable_framebuffer(self) -> bool {
        match self {
            BlendMode::Clear |
            BlendMode::SrcOver |
            BlendMode::DestOver |
            BlendMode::SrcIn |
            BlendMode::DestIn |
            BlendMode::SrcOut |
            BlendMode::DestOut |
            BlendMode::SrcAtop |
            BlendMode::DestAtop |
            BlendMode::Xor |
            BlendMode::Lighter |
            BlendMode::Copy => false,
            BlendMode::Lighten |
            BlendMode::Darken |
            BlendMode::Multiply |
            BlendMode::Screen |
            BlendMode::HardLight |
            BlendMode::Overlay |
            BlendMode::ColorDodge |
            BlendMode::ColorBurn |
            BlendMode::SoftLight |
            BlendMode::Difference |
            BlendMode::Exclusion |
            BlendMode::Hue |
            BlendMode::Saturation |
            BlendMode::Color |
            BlendMode::Luminosity => true,
        }
    }
}

pub(crate) trait ToCombineMode {
    fn to_combine_mode(self) -> i32;
}

impl ToCombineMode for PaintCompositeOp {
    fn to_combine_mode(self) -> i32 {
        match self {
            PaintCompositeOp::DestIn => COMBINER_CTRL_COLOR_COMBINE_DEST_IN,
            PaintCompositeOp::SrcIn => COMBINER_CTRL_COLOR_COMBINE_SRC_IN,
        }
    }
}
