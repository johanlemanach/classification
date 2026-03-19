import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app, preprocess_from_pil


class FakeModel:
    def __init__(self, probs):
        self.probs = np.array([probs], dtype=np.float32)
        self.last_input = None

    def predict(self, values, verbose=0):
        self.last_input = values
        return self.probs


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.upload_dir = self.base_path / "uploads"
        self.db_path = self.base_path / "feedback.db"
        self.fake_model = FakeModel([0.05, 0.8, 0.1, 0.05])
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(self.db_path),
                "UPLOAD_FOLDER": str(self.upload_dir),
                "ENABLE_MONITORING_DASHBOARD": False,
            },
            model_override=self.fake_model,
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_image_file(self, color=(255, 128, 64)):
        img = Image.new("RGB", (320, 180), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer

    def test_preprocess_keeps_pixels_in_255_range_for_model_rescaling_layer(self):
        img = Image.new("RGB", (320, 180), color=(255, 128, 64))

        result = preprocess_from_pil(img)

        self.assertEqual(result.shape, (1, 224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result.max()), 255.0, places=3)
        self.assertGreater(float(result.min()), 0.0)

    def test_predict_route_uses_resized_unscaled_input_and_renders_prediction(self):
        response = self.client.post(
            "/predict",
            data={"file": (self.make_image_file(), "sample.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Classe pr", response.data)
        self.assertIsNotNone(self.fake_model.last_input)
        self.assertEqual(self.fake_model.last_input.shape, (1, 224, 224, 3))
        self.assertAlmostEqual(float(self.fake_model.last_input.max()), 255.0, places=3)

    def test_feedback_route_persists_image_prediction_and_user_label(self):
        response = self.client.post(
            "/predict",
            data={"file": (self.make_image_file(), "sample.jpg")},
            content_type="multipart/form-data",
        )
        html = response.data.decode("utf-8")
        image_path_marker = 'name="image_path" value="'
        image_path = html.split(image_path_marker, 1)[1].split('"', 1)[0]

        feedback_response = self.client.post(
            "/feedback",
            data={
                "image_path": image_path,
                "predicted_label": "forest",
                "predicted_confidence": "0.8",
                "user_label": "meadow",
            },
        )

        self.assertEqual(feedback_response.status_code, 200)
        self.assertIn(b"Label utilisateur enregistr", feedback_response.data)

        db = sqlite3.connect(self.db_path)
        row = db.execute(
            "SELECT image_path, predicted_label, predicted_confidence, user_label FROM feedback_events"
        ).fetchone()
        db.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], image_path)
        self.assertEqual(row[1], "forest")
        self.assertEqual(row[3], "meadow")

    def test_metrics_endpoint_exposes_operational_counters(self):
        self.client.get("/health")
        self.client.post(
            "/predict",
            data={"file": (self.make_image_file(), "sample.jpg")},
            content_type="multipart/form-data",
        )

        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("requests_total", payload)
        self.assertEqual(payload["prediction_requests_total"], 1)


if __name__ == "__main__":
    unittest.main()
