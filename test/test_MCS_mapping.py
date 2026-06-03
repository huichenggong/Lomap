import unittest
from pathlib import Path
from lomap import mcs as lomap_mcs
from rdkit import Chem

base = Path(__file__).resolve().parent


def get_map_dict(mcs: lomap_mcs.MCS) -> dict:
    return {int(k): int(v) for k, v in (item.split(":") for item in mcs.all_atom_match_list().split(","))}


def get_map_set(mcs: lomap_mcs.MCS) -> set:
    map_AB_list = [(int(k), int(v)) for k, v in (item.split(":") for item in mcs.all_atom_match_list().split(","))]
    a_list = [a for a, b in map_AB_list]
    b_list = [b for a, b in map_AB_list]
    if not len(a_list) == len(set(a_list)):
        raise ValueError("Duplicate atom in A")
    if not len(b_list) == len(set(b_list)):
        raise ValueError("Duplicate atom in B")
    return set(map_AB_list)


class MyTestCase(unittest.TestCase):
    def test_ring_breaking(self):
        print("# ring breaking")
        sdf_dir = base / "schrodinger_sets/water_set/hsp90_woodhead"
        mol1 = Chem.SDMolSupplier(sdf_dir / "A01.sdf", removeHs=False)[0]
        mol2 = Chem.SDMolSupplier(sdf_dir / "A02.sdf", removeHs=False)[0]
        mol3 = Chem.SDMolSupplier(sdf_dir / "A03.sdf", removeHs=False)[0]
        mol4 = Chem.SDMolSupplier(sdf_dir / "A04.sdf", removeHs=False)[0]

        # ── A01 → A02 ──────────────────────────────────────────────────────────
        # A01 has a 5-membered ring (atoms 2,3,22,21,4); A02 has an open
        # isopropyl chain instead.  Ring atoms 2 and 22 map to chain atoms,
        # ring atom 3 (CH2) is left unmapped, breaking bonds 2-3 and 3-22.
        m12 = lomap_mcs.MCS(
            mol1, mol2,
            time=40, threed=True, max3d=1.5, element_change=True,
            seed="", shift=False, ring_breaking=True,
        )
        map_set = get_map_set(m12)
        self.assertSetEqual(
            map_set,
            {
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 12),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 16),
                (18, 17),
                (19, 18),
                (20, 19),
                (21, 20),
                (31, 29),
                (32, 30),
                (33, 31),
                (34, 32),
                (35, 33),
                (36, 34),
                (37, 35),
                (38, 36),
                (39, 37),
                (40, 38),
                (41, 39),
                # ring-breaking atoms
                (0, 0), (1, 2), (2, 1), (22, 21),
                # H on ring-breaking heavy atoms
                (23, 22), (24, 23), (25, 24),
                (26, 26), (27, 27), (28, 28),
                (42, 40),
            },
        )

        # Atom 3 (unmapped CH2) has two anchors: bonds 2-3 and 3-22.
        # Keep bond 2-3 (lowest mapped-atom index); break bond 3-22.
        # One anchor ensures the dummy partition function is separable.
        broken_moli, broken_molj = m12.broken_ring_bonds()
        self.assertEqual(broken_moli, [(3, 22)])
        self.assertEqual(broken_molj, [])

        # ── A03 → A04 ──────────────────────────────────────────────────────────
        # A03 has the same 5-membered ring as A01 (with O at position 22 instead
        # of N).  A04 has no ring there; O22 stays mapped (it's close in 3D to
        # A04's non-ring O21), and ring atom 3 is again left unmapped.
        m34 = lomap_mcs.MCS(
            mol3, mol4,
            time=40, threed=True, max3d=1.5, element_change=True,
            seed="", shift=False, ring_breaking=True,
        )
        map_set = get_map_set(m34)
        self.assertSetEqual(
            map_set,
            {
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 12),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 16),
                (18, 17),
                (19, 18),
                (20, 19),
                (21, 20),
                (31, 30),
                (32, 31),
                (33, 32),
                (34, 33),
                (35, 34),
                (36, 35),
                (37, 36),
                (38, 37),
                (39, 38),
                (40, 39),
                (41, 40),
                # ring-breaking atoms (O22 in ring maps to non-ring O21 in A04)
                (0, 2), (1, 0), (2, 1), (22, 21),
                # H on ring-breaking heavy atoms
                (23, 27), (24, 28), (25, 29),
                (26, 23), (27, 24), (28, 25),
            },
        )

        # Same ring structure as A01: atom 3 unmapped, keep bond 2-3, break 3-22.
        broken_moli, broken_molj = m34.broken_ring_bonds()
        self.assertEqual(broken_moli, [(3, 22)])
        self.assertEqual(broken_molj, [])

    def test_ring_easy_mapping(self):
        print("# no ring breaking")
        sdf_dir = base / "schrodinger_sets/water_set/hsp90_woodhead"
        mol1 = Chem.SDMolSupplier(sdf_dir / "A01.sdf", removeHs=False)[0]
        mol3 = Chem.SDMolSupplier(sdf_dir / "A03.sdf", removeHs=False)[0]
        mol2 = Chem.SDMolSupplier(sdf_dir / "A02.sdf", removeHs=False)[0]
        mol4 = Chem.SDMolSupplier(sdf_dir / "A04.sdf", removeHs=False)[0]

        m13 = lomap_mcs.MCS(mol1, mol3, time=40, threed=True, max3d=1.0, element_change=True, seed="", shift=False)
        map_set = get_map_set(m13)
        self.assertSetEqual(map_set, set([(i, i) for i in range(42)]))
        # no ring-breaking: both molecules have the same ring system
        broken_moli, broken_molj = m13.broken_ring_bonds()
        self.assertEqual(broken_moli, [])
        self.assertEqual(broken_molj, [])

        m24 = lomap_mcs.MCS(mol2, mol4, time=40, threed=True, max3d=1.5, element_change=True, seed="", shift=False)
        map_set = get_map_set(m24)
        self.assertSetEqual(
            map_set,
            set(
                [(0, 2), (1, 1), (2, 0)]
                + [(i, i) for i in range(3, 22)]
                + [(22, 27), (23, 28), (24, 29), (25, 26), (26, 23), (27, 24), (28, 25)]
                + [(i, i + 1) for i in range(29, 40)]
                + [(40, 22)]
            ),
        )
        broken_moli, broken_molj = m24.broken_ring_bonds()
        self.assertEqual(broken_moli, [])
        self.assertEqual(broken_molj, [])

    def test_ring_breaking_2(self):
        print("# ring breaking")
        sdf_dir = base / "ring_breaking_sample"
        mol1 = Chem.SDMolSupplier(sdf_dir / "6_6.sdf", removeHs=False)[0]
        mol2 = Chem.SDMolSupplier(sdf_dir / "6_5.sdf", removeHs=False)[0]
        mol3 = Chem.SDMolSupplier(sdf_dir / "6_6_OO.sdf", removeHs=False)[0]
        mol4 = Chem.SDMolSupplier(sdf_dir / "6_6_N.sdf", removeHs=False)[0]

        m12 = lomap_mcs.MCS(
            mol1, mol2,
            time=40, threed=True, max3d=1.5, element_change=True,
            seed="", shift=False, ring_breaking=True,
        )

        m13 = lomap_mcs.MCS(
            mol1, mol3,
            time=40, threed=True, max3d=1.5, element_change=True,
            seed="", shift=False, ring_breaking=True,
        )
        map_set = get_map_set(m13)
        self.assertSetEqual(
            map_set, {(14, 13), (17, 15), (2, 2), (11, 11), (7, 7), (16, 16), (15, 17), (3, 3), (12, 12), (21, 21), (8, 8), (19, 20), (18, 18), (4, 4), (5, 5), (0, 0), (9, 9), (1, 1), (10, 10), (6, 6)}
        )
        # 6_6_OO has an O-O bond (atoms 9-11 in _molj_noh = atoms 14-19 in full mol3).
        # Both O atoms are unmapped; each anchors to one real atom (0 and 7).
        # Break the internal O-O bond so each O is an independent dummy group
        # with exactly one anchor, keeping the real atoms' local environment intact.
        broken_moli, broken_molj = m13.broken_ring_bonds()
        self.assertEqual(broken_moli, [])
        self.assertEqual(broken_molj, [(14, 19)])

        m34 = lomap_mcs.MCS(
            mol3, mol4,
            time=40, threed=True, max3d=1.5, element_change=True,
            seed="", shift=False, ring_breaking=True,
        )
        self.assertTupleEqual(m34.broken_ring_bonds(), ([(14,19)],[]))

if __name__ == "__main__":
    unittest.main()
