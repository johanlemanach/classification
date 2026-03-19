# Satellite Image Classification App (E5 Project)

This project is a Flask-based application for satellite image classification, developed as part of a certification project (E5) for Johan. It classifies images into four categories: `desert`, `forest`, `meadow`, and `mountain` using a CNN model.

## Project Overview

- **Purpose:** Satellite image classification with built-in monitoring, alerting, and feedback mechanisms for MLOps demonstration.
- **Main Technologies:**
    - **Web Framework:** Flask
    - **Machine Learning:** Keras (TensorFlow backend), NumPy
    - **Image Processing:** Pillow (PIL)
    - **Database:** SQLite (for feedback storage)
    - **Monitoring:** `flask-monitoringdashboard`, custom `/metrics` endpoint, and logging with `RotatingFileHandler`.
- **Architecture:** 
    - A Flask web server handles image uploads and serves predictions.
    - A pre-trained Keras model (`app/models/final_cnn.keras`) performs inference.
    - User feedback is stored in a SQLite database (`app/instance/feedback.db`) to enable model improvement.
    - Logging is split into `app.log` (general events) and `alerts.log` (threshold violations and errors).

## Building and Running

### Prerequisites
- Python 3.8+
- Virtual environment recommended.

### Setup and Execution
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```
2. **Install dependencies:**
   ```bash
   pip install -r app/requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app/app.py
   ```
   The application will be available at `http://127.0.0.1:5000`.

4. **Run tests:**
   ```bash
   python -m unittest discover -s app/tests -v
   ```

## Development Conventions

### Inference & Preprocessing
- **Image Size:** The model requires a fixed input size of `224x224`.
- **Pixel Range:** The application must maintain pixel values in the `[0, 255]` range. The Keras model includes an internal `Rescaling(1/255)` layer, so **do not** divide pixel values by 255.0 during preprocessing.
- **Batching:** Inference expects a batch dimension, i.e., shape `(1, 224, 224, 3)`.

### Monitoring and Alerting
- **Metrics:** Exposed via the `/metrics` endpoint in JSON format.
- **Alerting Thresholds:**
    - Latency: `> 800 ms`
    - Confidence: `< 0.55`
    - Errors: HTTP 5xx responses.
- **Dashboard:** `flask-monitoringdashboard` is used for endpoint-level performance monitoring.

### Feedback Loop
- User corrections are saved to the `feedback_events` table in `app/instance/feedback.db`.
- These entries are intended to be used for future model retraining and evaluation.

### CI/CD
- GitHub Actions is configured to run tests on every Push and Pull Request (`.github/workflows/ci.yml`).

## Key Files
- `app/app.py`: Main Flask application logic, routes, and MLOps features.
- `app/tests/test_app.py`: Comprehensive test suite covering inference and feedback loops.
- `app/models/final_cnn.keras`: Pre-trained CNN model for image classification.
- `app/E5.md`: Detailed documentation of the project's MLOps and incident resolution aspects.
- `app/dashboard.cfg`: Configuration for `flask-monitoringdashboard`.
