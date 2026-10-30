import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "personal_lib" / "mmpretrain" / "models" / "backbones" / "sf_mamba"))

import sf_mamba_pq as sfpq  # noqa: E402


class SFMambaPqTest(unittest.TestCase):
    def test_exp_neg_pq_and_softplus_pq_accuracy(self) -> None:
        for dt in [0.0, 0.05, 0.25, 0.7, 1.5, 3.0, 6.0]:
            dt_q20 = int(round(dt * sfpq.PQ_Q20))
            got = sfpq.exp_neg_pq_q20(dt_q20) / float(sfpq.PQ_Q20)
            ref = math.exp(-dt)
            self.assertAlmostEqual(got, ref, delta=2e-4)

        for x in [-3.0, -1.0, -0.2, 0.0, 0.5, 2.0, 4.0]:
            x_q20 = int(round(x * sfpq.PQ_Q20))
            got_sp = sfpq.softplus_pq_q20(x_q20) / float(sfpq.PQ_Q20)
            ref_sp = math.log1p(math.exp(x))
            self.assertAlmostEqual(got_sp, ref_sp, delta=5e-4)

    def test_harmonic_selective_scan_matches_float_and_respects_fold_boundaries(self) -> None:
        rng = random.Random(2026)
        L = 24
        d_state = 8
        u_seq = [rng.uniform(-1.0, 1.0) for _ in range(L)]
        dt_raw = [rng.uniform(-1.5, 0.5) for _ in range(L)]
        B_seq = [[rng.uniform(-0.5, 0.5) for _ in range(d_state)] for _ in range(L)]
        C_seq = [[rng.uniform(-0.5, 0.5) for _ in range(d_state)] for _ in range(L)]
        # Batch-folded sample boundary at t = 12 (`indices[12] == 1`)
        indices = [1 if t in (0, 12) else (-1 if t in (11, 23) else 0) for t in range(L)]

        out_pq = sfpq.selective_scan_harmonic_pq(
            u_seq,
            dt_raw,
            B_seq,
            C_seq,
            D_val=0.75,
            delta_bias=0.1,
            position_indices=indices,
            d_state=d_state,
        )

        # Float reference scan with A_n = -(n + 1)
        h_ref = [0.0] * d_state
        out_ref = []
        for t in range(L):
            if t > 0 and indices[t] == 1:
                h_ref = [0.0] * d_state
            dt = math.log1p(math.exp(dt_raw[t] + 0.1))
            y = 0.75 * u_seq[t]
            for n in range(d_state):
                a_n = -(n + 1)
                h_ref[n] = math.exp(dt * a_n) * h_ref[n] + dt * u_seq[t] * B_seq[t][n]
                y += h_ref[n] * C_seq[t][n]
            out_ref.append(y)

        max_err = max(abs(a - b) for a, b in zip(out_pq, out_ref))
        self.assertLess(max_err, 2e-3)

    def test_closest_divisor_operator_precedence_fix(self) -> None:
        # For batch_size = 12, r = 3.0 -> int_r = 3 (exact divisor)
        self.assertEqual(sfpq.closest_divisor_fixed(12, 3.0), 3)
        # For batch_size = 12, r = 5.0 -> d=1 checks int_r+1 = 6 (12 % 6 == 0!)
        # Previously `12 % 5 + 1 == 3 != 0` missed 6 and returned 4 instead!
        self.assertEqual(sfpq.closest_divisor_fixed(12, 5.0), 6)
        # For r = 0.1 -> int_r clamped to 1 (avoids ZeroDivisionError on `batch_size % 0`)
        self.assertEqual(sfpq.closest_divisor_fixed(32, 0.1), 1)


if __name__ == "__main__":
    unittest.main()
