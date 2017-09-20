// pathfinder/client/src/svg-demo.ts
//
// Copyright © 2017 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

import * as glmatrix from 'gl-matrix';
import * as _ from 'lodash';

import {DemoAppController} from './app-controller';
import {AntialiasingStrategy, AntialiasingStrategyName, NoAAStrategy} from "./aa-strategy";
import {OrthographicCamera} from "./camera";
import {ECAAStrategy, ECAAMulticolorStrategy} from "./ecaa-strategy";
import {PathfinderMeshData} from "./meshes";
import {ShaderMap, ShaderProgramSource} from './shader-loader';
import { SVGLoader, BUILTIN_SVG_URI } from './svg-loader';
import {panic, unwrapNull} from './utils';
import {PathfinderDemoView, Timings} from './view';
import SSAAStrategy from "./ssaa-strategy";
import PathfinderBufferTexture from "./buffer-texture";

const parseColor = require('parse-color');

const SVG_NS: string = "http://www.w3.org/2000/svg";

const DEFAULT_FILE: string = 'tiger';

const ANTIALIASING_STRATEGIES: AntialiasingStrategyTable = {
    none: NoAAStrategy,
    ssaa: SSAAStrategy,
    ecaa: ECAAMulticolorStrategy,
};

interface AntialiasingStrategyTable {
    none: typeof NoAAStrategy;
    ssaa: typeof SSAAStrategy;
    ecaa: typeof ECAAStrategy;
}

class SVGDemoController extends DemoAppController<SVGDemoView> {
    start() {
        super.start();

        this.loader = new SVGLoader;

        this.loadInitialFile(this.builtinFileURI);
    }

    protected fileLoaded() {
        this.loader.loadFile(this.fileData);
        this.loader.partition().then(meshes => {
            this.meshes = meshes;
            this.meshesReceived();
        })
    }

    protected createView() {
        return new SVGDemoView(this,
                               unwrapNull(this.commonShaderSource),
                               unwrapNull(this.shaderSources));
    }

    protected readonly builtinFileURI: string = BUILTIN_SVG_URI;

    protected get defaultFile(): string {
        return DEFAULT_FILE;
    }

    private meshesReceived(): void {
        this.view.then(view => {
            view.uploadPathColors(1);
            view.uploadPathTransforms(1);
            view.attachMeshes([this.meshes]);

            view.camera.bounds = this.loader.bounds;
            view.camera.zoomToFit();
        })
    }

    loader: SVGLoader;

    private meshes: PathfinderMeshData;
}

class SVGDemoView extends PathfinderDemoView {
    constructor(appController: SVGDemoController,
                commonShaderSource: string,
                shaderSources: ShaderMap<ShaderProgramSource>) {
        super(commonShaderSource, shaderSources);

        this.appController = appController;

        this.camera = new OrthographicCamera(this.canvas, { scaleBounds: true });
        this.camera.onPan = () => this.setDirty();
        this.camera.onZoom = () => this.setDirty();
    }

    get destAllocatedSize(): glmatrix.vec2 {
        return glmatrix.vec2.fromValues(this.canvas.width, this.canvas.height);
    }

    get destFramebuffer(): WebGLFramebuffer | null {
        return null;
    }

    get destUsedSize(): glmatrix.vec2 {
        return this.destAllocatedSize;
    }

    protected pathColorsForObject(objectIndex: number): Uint8Array {
        const instances = this.appController.loader.pathInstances;
        const pathColors = new Uint8Array(4 * (instances.length + 1));

        for (let pathIndex = 0; pathIndex < instances.length; pathIndex++) {
            const startOffset = (pathIndex + 1) * 4;

            // Set color.
            const style = window.getComputedStyle(instances[pathIndex].element);
            const property = instances[pathIndex].stroke === 'fill' ? 'fill' : 'stroke';
            const color: number[] =
                style[property] === 'none' ? [0, 0, 0, 0] : parseColor(style[property]).rgba;
            pathColors.set(color.slice(0, 3), startOffset);
            pathColors[startOffset + 3] = color[3] * 255;
        }

        return pathColors;
    }

    protected pathTransformsForObject(objectIndex: number): Float32Array {
        const instances = this.appController.loader.pathInstances;
        const pathTransforms = new Float32Array(4 * (instances.length + 1));

        for (let pathIndex = 0; pathIndex < instances.length; pathIndex++) {
            // TODO(pcwalton): Set transform.
            const startOffset = (pathIndex + 1) * 4;
            pathTransforms.set([1, 1, 0, 0], startOffset);
        }

        return pathTransforms;
    }

    protected createAAStrategy(aaType: AntialiasingStrategyName,
                               aaLevel: number,
                               subpixelAA: boolean):
                               AntialiasingStrategy {
        return new (ANTIALIASING_STRATEGIES[aaType])(aaLevel, subpixelAA);
    }

    protected compositeIfNecessary(): void {}

    protected newTimingsReceived() {
        this.appController.newTimingsReceived(_.pick(this.lastTimings, ['rendering']));
    }

    protected usedSizeFactor: glmatrix.vec2 = glmatrix.vec2.fromValues(1.0, 1.0);

    protected get worldTransform() {
        const transform = glmatrix.mat4.create();
        const translation = this.camera.translation;
        glmatrix.mat4.translate(transform, transform, [-1.0, -1.0, 0.0]);
        glmatrix.mat4.scale(transform,
                            transform,
                            [2.0 / this.canvas.width, 2.0 / this.canvas.height, 1.0]);
        glmatrix.mat4.translate(transform, transform, [translation[0], translation[1], 0]);
        glmatrix.mat4.scale(transform, transform, [this.camera.scale, this.camera.scale, 1.0]);
        return transform;
    }

    protected get directCurveProgramName(): keyof ShaderMap<void> {
        return 'directCurve';
    }

    protected get directInteriorProgramName(): keyof ShaderMap<void> {
        return 'directInterior';
    }

    protected depthFunction: number = this.gl.GREATER;

    private appController: SVGDemoController;

    camera: OrthographicCamera;
}

function main() {
    const controller = new SVGDemoController;
    window.addEventListener('load', () => controller.start(), false);
}

main();
