import base64
import json
import unittest
from pathlib import Path

from src.services.omr.omr_service_v2 import process_image_dynamic


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class RealPhotoRegressionTest(unittest.TestCase):
    def test_processes_real_whatsapp_photo_fixture(self) -> None:
        image_base64 = base64.b64encode(
            (FIXTURES_DIR / "images" / "whatsapp-photo-2521190.jpeg").read_bytes()
        ).decode("ascii")

        response = process_image_dynamic(
            capture_id="capture-real-fixture",
            session_id="session-real-fixture",
            image_base64=image_base64,
            compiled_geometry_json=json.loads(
                (FIXTURES_DIR / "geometry" / "whatsapp-photo-2521190.json").read_text(
                    encoding="utf-8"
                )
            ),
            master_answers=None,
            threshold=0.5,
            delta=0.12,
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["registration"]["status"], "ok")
        self.assertEqual(response["registration"]["value"], "2521190")
        self.assertEqual(response["answers_numeric"], [3, 0, 1, 3, 4, 1, 0, 2, 1, 2])
        self.assertTrue(response["images"]["overlayBase64"])
        self.assertTrue(response["images"]["rectifiedBase64"])


if __name__ == "__main__":
    unittest.main()
