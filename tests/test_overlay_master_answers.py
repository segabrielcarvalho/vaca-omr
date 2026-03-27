import unittest

import numpy as np

from src.services.omr.omr_service_v2 import _draw_overlay


class OmrOverlayMasterAnswersTest(unittest.TestCase):
    def test_keeps_legacy_behavior_without_master_answers(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=1)],
            None,
        )

        self.assertTrue(np.array_equal(overlay[40, 60], np.array([0, 170, 0], dtype=np.uint8)))

    def test_marks_correct_answer_in_green_and_wrong_selection_in_orange(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=3)],
            [2],
        )

        self.assertTrue(np.array_equal(overlay[31, 60], np.array([0, 200, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(overlay[40, 140], np.array([0, 140, 255], dtype=np.uint8)))

    def test_marks_correct_selection_in_green_without_orange(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=2)],
            [3],
        )

        self.assertTrue(np.array_equal(overlay[40, 100], np.array([0, 170, 0], dtype=np.uint8)))

    def test_marks_correct_answer_in_green_when_student_leaves_blank(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=-1)],
            [4],
        )

        self.assertTrue(np.array_equal(overlay[31, 140], np.array([0, 200, 0], dtype=np.uint8)))

    def test_keeps_ambiguity_in_red_and_correct_answer_in_green(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=-2, best_index=0, second_index=1)],
            [4],
        )

        self.assertTrue(np.array_equal(overlay[32, 40], np.array([0, 0, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(overlay[32, 60], np.array([0, 0, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(overlay[31, 140], np.array([0, 200, 0], dtype=np.uint8)))

    def test_ignores_invalid_master_answer(self) -> None:
        overlay = _draw_overlay(
            self._blank_image(),
            [],
            [self._answer_row(decision=0)],
            [9],
        )

        self.assertTrue(np.array_equal(overlay[40, 40], np.array([0, 170, 0], dtype=np.uint8)))

    def _blank_image(self) -> np.ndarray:
        return np.full((80, 220, 3), 255, dtype=np.uint8)

    def _answer_row(
        self,
        *,
        decision: int,
        best_index: int = -1,
        second_index: int = -1,
    ) -> dict[str, object]:
        return {
            "bubbles": [
                (40, 40, 8),
                (60, 40, 8),
                (100, 40, 8),
                (140, 40, 8),
                (180, 40, 8),
            ],
            "decision": decision,
            "bestIndex": best_index,
            "secondIndex": second_index,
            "ratios": [0.0] * 5,
        }


if __name__ == "__main__":
    unittest.main()
