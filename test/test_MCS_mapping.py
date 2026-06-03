import unittest
from pathlib import Path
from lomap import mcs as lomap_mcs
from rdkit import Chem

base = Path(__file__).resolve().parent

def get_map_dict(mcs:lomap_mcs.MCS):
    map_dict = {int(k): int(v) for k, v in (item.split(":") for item in mcs.all_atom_match_list().split(","))}

def get_map_set(mcs:lomap_mcs.MCS):
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
        mol1 = Chem.SDMolSupplier(sdf_dir/'A01.sdf', removeHs=False)[0]
        mol2 = Chem.SDMolSupplier(sdf_dir/'A02.sdf', removeHs=False)[0]
        mol3 = Chem.SDMolSupplier(sdf_dir / 'A03.sdf', removeHs=False)[0]
        mol4 = Chem.SDMolSupplier(sdf_dir / 'A04.sdf', removeHs=False)[0]

        m12 = lomap_mcs.MCS(mol1, mol2, time=40, threed=True, max3d=1.5, element_change=True, seed='', shift=False)
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
                # ring breaking
                (1,2),(26, 26),(27, 27),(28,28),
                (2,1),
                (0,0),(23,22),(24,23),(25,24),
                (22,21),(42,40)
            })
        # bond breaking should happen in stateA 2-3 or 3-22

        m34 = lomap_mcs.MCS(mol3, mol4, time=40, threed=True, max3d=1.5, element_change=True, seed='', shift=False)
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
                # ring breaking
                (1,0),(26,23),(27,24),(28,25),
                (2,1),
                (0,2),(23,27),(24,28),(25,29),
            })
        # bond breaking should happen in stateA 2-3 or 3-22


    def test_ring_easy_mapping(self):
        print("# no ring breaking")
        sdf_dir = base / "schrodinger_sets/water_set/hsp90_woodhead"
        mol1 = Chem.SDMolSupplier(sdf_dir/'A01.sdf', removeHs=False)[0]
        mol3 = Chem.SDMolSupplier(sdf_dir/'A03.sdf', removeHs=False)[0]
        mol2 = Chem.SDMolSupplier(sdf_dir / 'A02.sdf', removeHs=False)[0]
        mol4 = Chem.SDMolSupplier(sdf_dir / 'A04.sdf', removeHs=False)[0]

        m13 = lomap_mcs.MCS(mol1, mol3, time=40, threed=True, max3d=1.0, element_change=True, seed='', shift=False)
        map_set = get_map_set(m13)
        self.assertSetEqual(map_set, set([(i,i) for i in range(42)]))

        m24 = lomap_mcs.MCS(mol2, mol4, time=40, threed=True, max3d=1.5, element_change=True, seed='', shift=False)
        map_set = get_map_set(m24)
        self.assertSetEqual(map_set,
                            set(
                                [(0,2),(1,1),(2,0)] + [(i,i) for i in range(3,22)]
                                + [(22, 27), (23, 28), (24, 29),
                                   (25, 26),
                                   (26, 23), (27, 24), (28, 25),]
                                + [(i, i+1) for i in range(29,40)] + [(40, 22)]
                            )
                            )

if __name__ == '__main__':
    unittest.main()
