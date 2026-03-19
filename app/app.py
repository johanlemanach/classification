import base64
import io
import logging
import sqlite3
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from flask import Flask, current_app, g, jsonify, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

try:
    import keras
except ImportError:  # pragma: no cover - handled at runtime when model loading is needed.
    keras = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
LOG_DIR = BASE_DIR / "logs"
INSTANCE_DIR = BASE_DIR / "instance"
DATABASE_PATH = INSTANCE_DIR / "feedback.db"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
CLASSES = ["desert", "forest", "meadow", "mountain"]
MODEL_PATH = BASE_DIR / "models" / "final_cnn.keras"
MODEL_INPUT_SIZE = (224, 224)
LOW_CONFIDENCE_THRESHOLD = 0.55
LATENCY_ALERT_THRESHOLD_MS = 800


class ModelUnavailableError(RuntimeError):
    """Raised when the Keras model cannot be loaded."""


def allowed_file(filename: str) -> bool:
    """Return True when the filename extension is supported."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_data_url(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 data URL."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def preprocess_from_pil(pil_img: Image.Image) -> np.ndarray:
    """Resize to the model input and keep pixel values in [0, 255]."""
    img = pil_img.convert("RGB").resize(MODEL_INPUT_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def load_model():
    """Load the Keras model lazily so tests can inject a fake model."""
    if keras is None:
        raise ModelUnavailableError(
            "Keras is not installed. Install dependencies from requirements.txt to run predictions."
        )
    return keras.saving.load_model(MODEL_PATH, compile=False)


def ensure_directories() -> None:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db_path = Path(current_app.config["DATABASE"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            user_label TEXT NOT NULL
        )
        """
    )
    db.commit()
    db.close()


def configure_logging(app: Flask) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    app.logger.handlers.clear()
    app.logger.setLevel(logging.INFO)

    app_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
    app_handler.setFormatter(formatter)
    app.logger.addHandler(app_handler)

    alert_logger = logging.getLogger("alerts")
    alert_logger.handlers.clear()
    alert_logger.setLevel(logging.WARNING)
    alert_handler = RotatingFileHandler(LOG_DIR / "alerts.log", maxBytes=500_000, backupCount=3)
    alert_handler.setFormatter(formatter)
    alert_logger.addHandler(alert_handler)

    if not app.testing:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        app.logger.addHandler(stream_handler)
        alert_logger.addHandler(stream_handler)

    app.extensions["alert_logger"] = alert_logger


def increment_metric(app: Flask, key: str, amount: int = 1) -> None:
    app.extensions["metrics"][key] = app.extensions["metrics"].get(key, 0) + amount


def update_average_latency(app: Flask, latency_ms: float) -> None:
    metrics = app.extensions["metrics"]
    previous_count = metrics.get("requests_total", 0) - 1
    previous_avg = metrics.get("avg_latency_ms", 0.0)
    if previous_count <= 0:
        metrics["avg_latency_ms"] = latency_ms
        return
    metrics["avg_latency_ms"] = ((previous_avg * previous_count) + latency_ms) / (previous_count + 1)


def save_upload(file_storage) -> Path:
    filename = secure_filename(file_storage.filename or "upload.jpg")
    extension = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    file_path = upload_folder / unique_name
    file_storage.save(file_path)
    return file_path


def check_prediction_alerts(app: Flask, confidence: float) -> None:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        increment_metric(app, "low_confidence_predictions_total")
        app.extensions["alert_logger"].warning(
            "Prediction confidence %.3f below threshold %.2f",
            confidence,
            LOW_CONFIDENCE_THRESHOLD,
        )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def create_app(test_config=None, model_override=None) -> Flask:
    ensure_directories()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="dev",
        DATABASE=str(DATABASE_PATH),
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        MODEL_LOADER=load_model,
        MODEL=None,
        TESTING=False,
    )
    if test_config:
        app.config.update(test_config)
    if model_override is not None:
        app.config["MODEL"] = model_override

    configure_logging(app)
    app.extensions["metrics"] = {
        "requests_total": 0,
        "prediction_requests_total": 0,
        "feedback_events_total": 0,
        "errors_total": 0,
        "low_confidence_predictions_total": 0,
        "avg_latency_ms": 0.0,
        "last_latency_ms": 0.0,
    }

    if app.config.get("ENABLE_MONITORING_DASHBOARD", True):
        try:
            import flask_monitoringdashboard as dashboard

            dashboard.config.init_from(file=str(BASE_DIR / "dashboard.cfg"))
            dashboard.bind(app)
            app.logger.info("flask-monitoringdashboard enabled")
        except ImportError:
            app.logger.info("flask-monitoringdashboard not installed; JSON metrics endpoint remains available")
        except Exception as exc:  # pragma: no cover - defensive logging for optional dependency.
            app.logger.warning("Unable to initialize flask-monitoringdashboard: %s", exc)

    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()

    @app.before_request
    def before_request():
        g.request_started_at = time.perf_counter()

    @app.after_request
    def after_request(response):
        latency_ms = round((time.perf_counter() - g.request_started_at) * 1000, 2)
        increment_metric(app, "requests_total")
        app.extensions["metrics"]["last_latency_ms"] = latency_ms
        update_average_latency(app, latency_ms)
        app.logger.info(
            "http_request method=%s path=%s status=%s latency_ms=%.2f remote_addr=%s",
            request.method,
            request.path,
            response.status_code,
            latency_ms,
            request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        if response.status_code >= 500:
            increment_metric(app, "errors_total")
            app.extensions["alert_logger"].error(
                "HTTP 5xx detected for path=%s status=%s",
                request.path,
                response.status_code,
            )
        if latency_ms > LATENCY_ALERT_THRESHOLD_MS:
            app.extensions["alert_logger"].warning(
                "Latency threshold exceeded path=%s latency_ms=%.2f threshold_ms=%s",
                request.path,
                latency_ms,
                LATENCY_ALERT_THRESHOLD_MS,
            )
        return response

    @app.route("/", methods=["GET"])
    def index():
        return render_template("upload.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "" or not allowed_file(secure_filename(file.filename)):
            return redirect(url_for("index"))

        saved_path = save_upload(file)
        pil_img = Image.open(saved_path)
        img_array = preprocess_from_pil(pil_img)

        model = app.config["MODEL"]
        if model is None:
            model = app.config["MODEL_LOADER"]()
            app.config["MODEL"] = model

        probs = model.predict(img_array, verbose=0)[0]
        cls_idx = int(np.argmax(probs))
        label = CLASSES[cls_idx]
        conf = float(probs[cls_idx])
        increment_metric(app, "prediction_requests_total")
        check_prediction_alerts(app, conf)

        app.logger.info(
            "prediction_completed image_path=%s predicted_label=%s confidence=%.3f",
            display_path(saved_path),
            label,
            conf,
        )

        return render_template(
            "result.html",
            image_data_url=to_data_url(pil_img, fmt="JPEG"),
            predicted_label=label,
            confidence=conf,
            classes=CLASSES,
            image_path=display_path(saved_path),
        )

    @app.route("/feedback", methods=["POST"])
    def feedback():
        image_path = request.form.get("image_path", "").strip()
        predicted_label = request.form.get("predicted_label", "").strip()
        predicted_confidence = request.form.get("predicted_confidence", "0").strip()
        user_label = request.form.get("user_label", "").strip()

        if not image_path or predicted_label not in CLASSES or user_label not in CLASSES:
            return redirect(url_for("index"))

        try:
            confidence = float(predicted_confidence)
        except ValueError:
            return redirect(url_for("index"))

        db = get_db()
        db.execute(
            """
            INSERT INTO feedback_events (image_path, predicted_label, predicted_confidence, user_label)
            VALUES (?, ?, ?, ?)
            """,
            (image_path, predicted_label, confidence, user_label),
        )
        db.commit()
        increment_metric(app, "feedback_events_total")
        app.logger.info(
            "feedback_recorded image_path=%s predicted_label=%s user_label=%s",
            image_path,
            predicted_label,
            user_label,
        )
        return render_template("feedback_ok.html", user_label=user_label)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/metrics", methods=["GET"])
    def metrics():
        return jsonify(app.extensions["metrics"]), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
