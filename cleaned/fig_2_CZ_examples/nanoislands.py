import czone as cz
import numpy as np
from czone.generator import Generator
from czone.scene import Scene
from czone.transform import Rotation, Translation, rot_v, rot_vtv
from czone.volume import Volume, makeRectPrism, snap_plane_near_point
from pymatgen.core import Structure


def make_nanoisland(base_gen, radius, height, pos, theta):
    Au_island_gen = base_gen.from_generator()
    zone_axis = Au_island_gen.voxel.reciprocal_bases @ np.ones((3, 1))

    ## Rotate 111 axis to +Z, and then rotate about to get standard alignment of lattice
    rot_111_001 = Rotation(matrix=rot_vtv(v=zone_axis.T, vt=np.array(([0, 0, 1]))))

    rot_in_plane = Rotation(matrix=rot_v([0, 0, 1], -np.pi / 4))

    Au_island_gen.transform(rot_111_001)
    Au_island_gen.transform(rot_in_plane)

    ## Get bottom and top surfaces of nanoisland
    d_111 = 4.07825 / np.sqrt(3)

    bottom_plane = snap_plane_near_point(np.zeros((3)), Au_island_gen, (-1, -1, -1))
    top_plane = snap_plane_near_point(np.zeros((3)), Au_island_gen, (-1, -1, -1))
    top_plane.flip_orientation()
    top_plane = Translation(shift=[0, 0, height * d_111]).applyTransformation_alg(top_plane)

    ## Get sides of nanoisland
    d = (15 / 32.5) * radius * (1 / np.sqrt(2))  # Heuristic value to get island aspect ratio looking right
    side_plane_0 = snap_plane_near_point(np.array([d, d, height]), Au_island_gen, (1, 1, -1))
    side_plane_1 = snap_plane_near_point(np.array([-d, d, height]), Au_island_gen, (-1, 1, 1))
    side_plane_2 = snap_plane_near_point(np.array([d, -d, height]), Au_island_gen, (1, -1, 1))

    side_planes = [side_plane_0, side_plane_1, side_plane_2]

    ## Create nanoisland, then rotate and move to its desired location/alignment
    Au_island = Volume(alg_objects=[bottom_plane, top_plane] + side_planes, generator=Au_island_gen)

    Au_island.transform(Rotation(matrix=rot_v([0, 0, 1], theta)))
    Au_island.transform(Translation(shift=np.array([pos[0], pos[1], 15])))

    return Au_island


def generate_nanoisland_example():
    """
    Manually create a set of nanoislands on a MoS2 substrate.
    """
    ## Create MoS2 bilayer
    mos2_crystal = Structure.from_file("MoS2.cif")
    mos2_gen = Generator(structure=mos2_crystal)

    mos2_bounds = makeRectPrism(250, 250, 13.5)
    mos2_sheet = Volume(points=mos2_bounds, generator=mos2_gen)

    ## Define size/shapes/orientations of nanoislands
    radii = [32.5, 20, 40, 27, 35, 50]
    heights = [5, 3, 6, 7, 7, 10]
    thetas = [np.pi * x for x in [-1 / 6, 1 / 3, -1 / 12, 1 / 6, 0, 5 / 6]]
    positions = [(125, 125), (115, 40), (200, 50), (200, 210), (75, 200), (50, 140)]

    islands = []

    Au_gen = cz.generator.Generator.from_spacegroup(
        Z=[79],
        coords=np.array([[0, 0, 0]]),
        cellDims=4.07825 * np.ones((3,)),
        cellAngs=90 * np.ones((3,)),
        sgn=225,
    )

    ## Make nanoislands
    for r, h, t, pos in zip(radii, heights, thetas, positions):
        Au_island = make_nanoisland(Au_gen, r, h, pos, t)
        Au_island.priority = 1
        islands.append(Au_island)

    ## Combine objects into scene and write to file
    nanoisland_scene = Scene(bounds=np.array([[0, 0, 0], [250, 250, 50]]), objects=[mos2_sheet] + islands)

    nanoisland_scene.populate()
    nanoisland_scene.to_file("nanoislands.xyz")


if __name__ == "__main__":
    generate_nanoisland_example()
