import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.services.omr.debug_dump import write_omr_debug_dump


class WriteOmrDebugDumpTest(unittest.TestCase):
    def test_writes_artifacts_and_metadata_in_all_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_a = Path(temp_dir) / "tmp-root"
            root_b = Path(temp_dir) / "host-root"
            with patch(
                "src.services.omr.debug_dump.DEBUG_ROOT_DIRS",
                (root_a, root_b),
            ):
                dump_dirs = [
                    Path(dump_dir)
                    for dump_dir in write_omr_debug_dump(
                        capture_id="capture:1",
                        session_id="session/1",
                        artifacts={
                            "00_input.jpg": np.zeros((16, 16, 3), dtype=np.uint8),
                            "01_gray.jpg": np.zeros((16, 16), dtype=np.uint8),
                            "07_overlay_final.jpg": None,
                        },
                        metadata={
                            "arucoDetected": [0, 1, 2, 3],
                            "targetAnchorCentersPx": {"0": np.array([10.0, 20.0])},
                        },
                    )
                ]

            self.assertEqual(len(dump_dirs), 2)
            for dump_dir in dump_dirs:
                self.assertTrue(dump_dir.exists())
                self.assertTrue((dump_dir / "00_input.jpg").exists())
                self.assertTrue((dump_dir / "01_gray.jpg").exists())
                self.assertFalse((dump_dir / "07_overlay_final.jpg").exists())
                self.assertIn("session_1", dump_dir.as_posix())
                self.assertIn("capture_1__", dump_dir.name)

                payload = json.loads(
                    (dump_dir / "metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["captureId"], "capture:1")
                self.assertEqual(payload["sessionId"], "session/1")
                self.assertEqual(payload["arucoDetected"], [0, 1, 2, 3])
                self.assertEqual(payload["targetAnchorCentersPx"]["0"], [10.0, 20.0])

    def test_creates_metadata_even_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("src.services.omr.debug_dump.DEBUG_ROOT_DIRS", (root,)):
                dump_dir = Path(
                    write_omr_debug_dump(
                        capture_id=None,
                        session_id=None,
                        artifacts={
                            "00_input.jpg": None,
                        },
                        metadata={"success": False},
                    )[0]
                )

            self.assertTrue((dump_dir / "metadata.json").exists())
            self.assertEqual(dump_dir.parent.name, "session-unknown")


if __name__ == "__main__":
    unittest.main()
