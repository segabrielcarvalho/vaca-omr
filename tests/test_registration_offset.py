import unittest
from unittest.mock import patch

import numpy as np

from src.services.omr.omr_service_v2 import _estimate_registration_offset


class RegistrationOffsetTest(unittest.TestCase):
    def test_skips_offset_search_when_nominal_grid_has_strong_signal(self) -> None:
        registration = {
            "startXmm": 16.78,
            "startYmm": 92.66,
            "columns": 7,
            "rows": 10,
            "colGapMm": 8,
            "rowGapMm": 6,
        }
        page = {
            "x_scale": 10,
            "y_scale": 10,
        }
        gray = np.full((3000, 2100), 255, dtype=np.uint8)

        with patch(
            "src.services.omr.omr_service_v2._bubble_presence_score",
            return_value=0.2,
        ) as score:
            offset_x, offset_y, refined = _estimate_registration_offset(
                gray,
                registration,
                page,
                radius=20,
            )

        self.assertEqual((offset_x, offset_y, refined), (0, 0, False))
        self.assertEqual(score.call_count, 70)


if __name__ == "__main__":
    unittest.main()
