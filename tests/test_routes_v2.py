import unittest
from unittest.mock import patch

from src.routes.routes_v2 import OmrProcessRequest, omr_process


class OmrRoutesV2Test(unittest.IsolatedAsyncioTestCase):
    async def test_process_accepts_capture_and_session_ids(self) -> None:
        payload = OmrProcessRequest(
            captureId="capture-1",
            sessionId="session-1",
            imageBase64="Zm9v",
            compiledGeometryJson={"questions": {"questionCount": 1}},
            masterAnswers=[2],
        )

        with patch("src.routes.routes_v2.process_image_dynamic", return_value={"success": True}) as process:
            response = await omr_process(payload)

        process.assert_called_once_with(
            capture_id="capture-1",
            session_id="session-1",
            image_base64="Zm9v",
            compiled_geometry_json={"questions": {"questionCount": 1}},
            master_answers=[2],
            threshold=0.5,
            delta=0.12,
        )
        self.assertEqual(response.status_code, 200)

    async def test_process_keeps_backward_compatibility_without_ids(self) -> None:
        payload = OmrProcessRequest(
            imageBase64="Zm9v",
            compiledGeometryJson={"questions": {"questionCount": 1}},
        )

        with patch("src.routes.routes_v2.process_image_dynamic", return_value={"success": True}) as process:
            response = await omr_process(payload)

        process.assert_called_once_with(
            capture_id=None,
            session_id=None,
            image_base64="Zm9v",
            compiled_geometry_json={"questions": {"questionCount": 1}},
            master_answers=None,
            threshold=0.5,
            delta=0.12,
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
