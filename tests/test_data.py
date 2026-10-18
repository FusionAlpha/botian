import unittest

import numpy as np

from bo_t.data import load_frozen


class FrozenDataTests(unittest.TestCase):
    def test_verified_arrays_cover_all_scans(self):
        counts = dict(es1d_beam=50, umeda_ring_beam=117, signed_em=500,
                      hh_ring_beam=117, firehose=46, muschietti_beam=131,
                      loss_cone_mirror=100)
        for name, expected in counts.items():
            with self.subTest(case=name):
                raw, selected, evidence = load_frozen(name)
                self.assertEqual(len(np.unique(raw[:, 0])), expected)
                self.assertEqual(evidence["numerical_recomputation"], False)
                self.assertGreater(len(selected), 0)
                self.assertLess(len(selected), len(raw))


if __name__ == "__main__":
    unittest.main()
