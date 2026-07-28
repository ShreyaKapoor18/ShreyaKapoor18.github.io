"""Render neural fiber tractography streamlines in Blender.

Built for bundle files in the style of the ISMRM 2015 Tractography
Challenge ground-truth data (one .tck/.trk file per fiber bundle), but
works with any single .trk/.tck tractogram too.

Usage (run from a terminal, not Blender's GUI console):

    blender --background --python blender_tractography_render.py -- \\
        --input path/to/bundles_dir --output render.png

`--input` accepts either one .trk/.tck file or a directory of them.

Requires nibabel inside Blender's bundled Python (numpy ships with
Blender already):

    <blender_dir>/python/bin/python3.x -m pip install nibabel
"""

import argparse
import colorsys
import math
import os
import random
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "nibabel is required. Install it into Blender's Python with:\n"
        "  <blender_dir>/python/bin/python3.x -m pip install nibabel"
    ) from exc


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help=".trk/.tck file, or a directory of bundle files")
    parser.add_argument("--output", default="tractography_render.png")
    parser.add_argument("--max-streamlines", type=int, default=3000,
                         help="Subsample per bundle for scene/render performance")
    parser.add_argument("--radius", type=float, default=0.3,
                         help="Fiber tube radius, in the streamline file's coordinate units (usually mm)")
    parser.add_argument("--segments", type=int, default=6,
                         help="Sides per fiber tube cross-section")
    parser.add_argument("--color-mode", choices=["direction", "bundle"], default="direction",
                         help="direction = per-point DEC color, bundle = one flat color per file")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--background", choices=["white", "black"], default="black")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def find_bundle_files(input_path):
    if os.path.isdir(input_path):
        exts = (".trk", ".tck")
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(exts)
        )
        if not files:
            raise FileNotFoundError(f"No .trk/.tck files found in {input_path}")
        return files
    return [input_path]


def load_streamlines(path, max_streamlines, rng):
    tractogram = nib.streamlines.load(path)
    streamlines = list(tractogram.streamlines)
    if max_streamlines and len(streamlines) > max_streamlines:
        streamlines = rng.sample(streamlines, max_streamlines)
    return streamlines


def bundle_color(index, total):
    hue = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (r, g, b, 1.0)


def tube_ring(center, tangent, radius, segments, up_hint=Vector((0.0, 0.0, 1.0))):
    tangent = Vector(tangent)
    if tangent.length < 1e-8:
        tangent = Vector((0.0, 0.0, 1.0))
    else:
        tangent = tangent.normalized()
    hint = up_hint if abs(tangent.dot(up_hint)) < 0.99 else Vector((1.0, 0.0, 0.0))
    side = tangent.cross(hint).normalized()
    up = side.cross(tangent).normalized()
    center = Vector(center)
    return [
        center + radius * (math.cos(theta) * side + math.sin(theta) * up)
        for theta in (2 * math.pi * i / segments for i in range(segments))
    ]


def build_bundle_mesh(name, streamlines, radius, segments, color_mode, bundle_rgba):
    bm = bmesh.new()
    color_layer = bm.loops.layers.color.new("tract_color")

    for streamline in streamlines:
        if len(streamline) < 2:
            continue
        pts = np.asarray(streamline, dtype=np.float64)
        tangents = np.gradient(pts, axis=0)

        rings, colors = [], []
        for point, tangent in zip(pts, tangents):
            rings.append(tube_ring(point, tangent, radius, segments))
            if color_mode == "bundle":
                colors.append(bundle_rgba)
            else:
                t = tangent / (np.linalg.norm(tangent) + 1e-12)
                colors.append((abs(t[0]), abs(t[1]), abs(t[2]), 1.0))

        ring_verts = [[bm.verts.new(v) for v in ring] for ring in rings]

        for i in range(len(ring_verts) - 1):
            ring_a, ring_b = ring_verts[i], ring_verts[i + 1]
            color_a, color_b = colors[i], colors[i + 1]
            for j in range(segments):
                j2 = (j + 1) % segments
                face = bm.faces.new((ring_a[j], ring_a[j2], ring_b[j2], ring_b[j]))
                for loop in face.loops:
                    loop[color_layer] = color_a if loop.vert in (ring_a[j], ring_a[j2]) else color_b

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_material():
    mat = bpy.data.materials.new("TractMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "tract_color"

    bsdf.inputs["Roughness"].default_value = 0.35
    # Socket renamed "Specular" -> "Specular IOR Level" in Blender 4.0.
    specular_input = bsdf.inputs.get("Specular") or bsdf.inputs.get("Specular IOR Level")
    if specular_input is not None:
        specular_input.default_value = 0.5

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    attr.location = (-300, 0)
    bsdf.location = (0, 0)
    output.location = (300, 0)
    return mat


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def setup_world(background):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0) if background == "black" else (0.95, 0.95, 0.95, 1.0)


def compute_bounds(objects):
    min_co = Vector((math.inf, math.inf, math.inf))
    max_co = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            min_co = Vector(min(a, b) for a, b in zip(min_co, world_co))
            max_co = Vector(max(a, b) for a, b in zip(max_co, world_co))
    return min_co, max_co


def setup_camera_and_lights(objects):
    min_co, max_co = compute_bounds(objects)
    center = (min_co + max_co) / 2
    radius = (max_co - min_co).length / 2 or 1.0

    cam_data = bpy.data.cameras.new("TractCamera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("TractCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    direction = Vector((1.0, -1.2, 0.7)).normalized()
    cam_obj.location = center + direction * radius * 2.8
    cam_obj.rotation_euler = (center - cam_obj.location).normalized().to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam_obj

    sun_data = bpy.data.lights.new("KeyLight", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("KeyLight", sun_data)
    sun_obj.rotation_euler = (math.radians(50), 0, math.radians(35))
    bpy.context.collection.objects.link(sun_obj)

    fill_data = bpy.data.lights.new("FillLight", type="AREA")
    fill_data.energy = 400.0
    fill_data.size = radius * 2
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = center + Vector((-radius * 2, -radius * 2, radius))
    bpy.context.collection.objects.link(fill_obj)


def configure_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"
    except AttributeError:
        pass
    scene.cycles.samples = args.samples
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.filepath = args.output
    scene.render.image_settings.file_format = "PNG"


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    clear_scene()

    bundle_paths = find_bundle_files(args.input)
    objects = []
    for i, path in enumerate(bundle_paths):
        streamlines = load_streamlines(path, args.max_streamlines, rng)
        name = os.path.splitext(os.path.basename(path))[0]
        obj = build_bundle_mesh(
            name, streamlines, args.radius, args.segments,
            args.color_mode, bundle_color(i, len(bundle_paths)),
        )
        obj.data.materials.append(build_material())
        objects.append(obj)
        print(f"[{name}] built {len(streamlines)} streamlines")

    setup_world(args.background)
    setup_camera_and_lights(objects)
    configure_render(args)

    bpy.ops.render.render(write_still=True)
    print(f"Saved render to {args.output}")


if __name__ == "__main__":
    main()
