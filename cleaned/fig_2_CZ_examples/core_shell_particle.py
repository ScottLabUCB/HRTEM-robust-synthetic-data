import czone as cz
import numpy as np
from czone.generator import Generator
from czone.transform import Rotation, Translation, rot_v
from czone.volume import MultiVolume, Volume, snap_plane_near_point
from pymatgen.core import Structure


def generate_core_shell_example():
    ## Load Mn3O4 and Co3O4 structures
    mn_crystal = Structure.from_file("Mn3O4_mp-18759_conventional_standard.cif")
    co_crystal = Structure.from_file("Co3O4_mp-18748_conventional_standard.cif")

    # correct lattice mismatch
    co_100 = np.linalg.norm(co_crystal.lattice.matrix @ np.array([1, 0, 0]))
    mn_110 = np.linalg.norm(mn_crystal.lattice.matrix @ np.array([1, 1, 0]))
    mn_crystal.apply_strain([co_100 / mn_110 - 1, co_100 / mn_110 - 1, 0])

    ### Make Co3O4 core
    co_gen = Generator(structure=co_crystal)

    N_uc = 6  # stretch N unit cells from center
    co_a = co_crystal.lattice.a
    vecs_100 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    planes_100 = []
    for v in vecs_100:
        planes_100.append(snap_plane_near_point(N_uc * co_a * v, co_gen, tuple(v)))

    co_core = Volume(alg_objects=planes_100, generator=co_gen)

    ### Make one Mn3O4 grain
    mn_gen = Generator(structure=mn_crystal)

    # rotate 45 degrees and place on top of Co core
    r = Rotation(matrix=rot_v(np.array([0, 0, 1]), np.pi / 4))
    t = Translation(np.array([0, 0, N_uc * co_a]))
    mn_gen.transform(r)
    mn_gen.transform(t)

    # define top and bottom 100 surfaces
    surface_point = np.array([0, 0, N_uc * co_a])
    mn_c = mn_crystal.lattice.c
    mn_bot = snap_plane_near_point(surface_point + 1, mn_gen, (0, 0, -4))
    mn_top = snap_plane_near_point(surface_point + 6 * mn_c * np.array([0, 0, 1]), mn_gen, (0, 0, 1))

    # strain the grain
    sf = 0.875225
    mn_gen.strain_field = cz.transform.strain.HStrain(matrix=(1, 1, sf))

    # define 112 facets
    side_vs_112 = [(1, 1, -2), (1, -1, -2), (-1, 1, -2), (-1, -1, -2)]
    sides_112 = []
    for s in side_vs_112:
        tmp_point = np.array([1, 1, 0]) * np.sign(s) * co_a
        tmp_point = np.array([0, 0, 0]) + np.array([1, 1, 0]) * np.sign(s) * co_a * 0.1
        tmp_plane = snap_plane_near_point(tmp_point, mn_gen, s, mode="nearest")
        tmp_plane.point += tmp_plane.normal
        sides_112.append(tmp_plane)

    # define 101 facets
    mn_a = mn_crystal.lattice.a
    side_vs_101 = [(1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1)]
    sides_101 = []
    for s in side_vs_101:
        tmp_point = np.array([1, 1, 0]) * np.sign(s) * 12 * mn_a * sf + mn_top.point
        tmp_plane = snap_plane_near_point(tmp_point, mn_gen, s, mode="nearest")
        sides_101.append(tmp_plane)

    # create volume representing grain
    mn_vols = [mn_bot, mn_top] + sides_112 + sides_101
    mn_grain = Volume(alg_objects=mn_vols, generator=mn_gen)
    mn_grain.priority = 1

    ### Rotate original grain to make 5 other grains from +z shell grain
    mn_grains = [mn_grain]

    # get +x, -z, -x
    for theta in [np.pi / 2, np.pi, -np.pi / 2]:
        rot = Rotation(rot_v(np.array([0, 1, 0]), theta))
        tmp_grain = mn_grain.from_volume(transformation=[rot])
        mn_grains.append(tmp_grain)

    # get +-y
    for theta in [np.pi / 2, -np.pi / 2]:
        rot = Rotation(rot_v(np.array([1, 0, 0]), theta))
        tmp_grain = mn_grain.from_volume(transformation=[rot])
        mn_grains.append(tmp_grain)

    grain_precedence = [0, 1, 0, 1, 2, 2]
    for i, grain in enumerate(mn_grains):
        grain.priority += grain_precedence[i]

    # make final core-shell NP as multivolume and save to file
    core_shell_NP = MultiVolume([co_core] + mn_grains)
    core_shell_NP.populate_atoms()
    core_shell_NP.to_file("core_shell_NP.xyz")


if __name__ == "__main__":
    generate_core_shell_example()
