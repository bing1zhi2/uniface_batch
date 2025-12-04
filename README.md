# uniface-batch

## test use dcu device k100,use dtk 2504
## modify code in yolv5 to support batch inference


## see [UniFace: All-in-One Face Analysis Library](https://github.com/yakhyo/uniface)
**UniFace** is a lightweight, production-ready face analysis library built on ONNX Runtime. It provides high-performance face detection, recognition, landmark detection, and attribute analysis with hardware acceleration support across platforms.


## Installation


### Install from Source

use docker img: image.sourcefind.cn:5000/dcu/admin/base/migraphx:5.0.0-ubuntu22.04-dtk25.04.1-py3.10

```bash
git clone https://github.com/bing1zhi2/uniface_batch
cd uniface_batch
pip install -e .
```

---

## Quick Start

## demo

``` python

import cv2
from uniface import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights
from uniface import ArcFace
from uniface.constants import ArcFaceWeights
from uniface import Landmark106



# Real-time detection (recommended)
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    conf_thresh=0.6,
    nms_thresh=0.5
)

# High accuracy (ResNet50 backbone)
recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)

# landmarker = Landmark106()




# Load image
# image = cv2.imread("/root/develop/sdkdev/uniface_sdk/uniface/assets/test.jpg")
image = cv2.imread("/root/develop/sdkdev/uniface_sdk/uniface/assets/test_images/image1.jpg")


# faces = detector.detect(image)

# # Process results
# for face in faces:
#     bbox = face['bbox']  # [x1, y1, x2, y2]
#     confidence = face['confidence']
#     landmarks = face['landmarks']  # 5-point landmarks
#     print(f"Face detected with confidence: {confidence:.2f}")
#     # Extract embedding
#     embedding = recognizer.get_normalized_embedding(image, landmarks)
#     print(embedding)

all_img_faces = detector.detect_batch([image,image])

# Process results
for faces in all_img_faces:
    for face in faces:
        bbox = face['bbox']  # [x1, y1, x2, y2]
        confidence = face['confidence']
        landmarks = face['landmarks']  # 5-point landmarks
        print(f"Face detected with confidence: {confidence:.2f}")
        # Extract embedding
        embedding = recognizer.get_normalized_embedding(image, landmarks)
        # print(embedding)
```

### Face Detection

```python
import cv2
from uniface import RetinaFace

# Initialize detector
detector = RetinaFace()

# Load image
image = cv2.imread("image.jpg")

# Detect faces
faces = detector.detect(image)

# Process results
for face in faces:
    bbox = face['bbox']  # [x1, y1, x2, y2]
    confidence = face['confidence']
    landmarks = face['landmarks']  # 5-point landmarks
    print(f"Face detected with confidence: {confidence:.2f}")
```

### Face Recognition

```python
from uniface import ArcFace, RetinaFace
from uniface import compute_similarity

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()

# Detect and extract embeddings
faces1 = detector.detect(image1)
faces2 = detector.detect(image2)

embedding1 = recognizer.get_normalized_embedding(image1, faces1[0]['landmarks'])
embedding2 = recognizer.get_normalized_embedding(image2, faces2[0]['landmarks'])

# Compare faces
similarity = compute_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")
```

### Facial Landmarks

```python
from uniface import RetinaFace, Landmark106

detector = RetinaFace()
landmarker = Landmark106()

faces = detector.detect(image)
landmarks = landmarker.get_landmarks(image, faces[0]['bbox'])
# Returns 106 (x, y) landmark points
```

### Age & Gender Detection

```python
from uniface import RetinaFace, AgeGender

detector = RetinaFace()
age_gender = AgeGender()

faces = detector.detect(image)
gender_id, age = age_gender.predict(image, faces[0]['bbox'])
gender = 'Female' if gender_id == 0 else 'Male'
print(f"{gender}, {age} years old")
```

---

## Documentation

- [**QUICKSTART.md**](QUICKSTART.md) - 5-minute getting started guide
- [**MODELS.md**](MODELS.md) - Model zoo, benchmarks, and selection guide
- [**Examples**](examples/) - Jupyter notebooks with detailed examples

---

## API Overview

### Factory Functions (Recommended)

```python
from uniface.detection import RetinaFace, SCRFD
from uniface.recognition import ArcFace
from uniface.landmark import Landmark106

# Create detector with default settings
detector = RetinaFace()

# Create with custom config
detector = SCRFD(
    model_name='scrfd_10g_kps',
    conf_thresh=0.8,
    input_size=(640, 640)
)

# Recognition and landmarks
recognizer = ArcFace()
landmarker = Landmark106()
```

### Direct Model Instantiation

```python
from uniface import RetinaFace, SCRFD, YOLOv5Face, ArcFace, MobileFace, SphereFace
from uniface.constants import RetinaFaceWeights, YOLOv5FaceWeights

# Detection
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,
    conf_thresh=0.5,
    nms_thresh=0.4
)

# YOLOv5-Face detection
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    conf_thresh=0.6,
    nms_thresh=0.5
)

# Recognition
recognizer = ArcFace()  # Uses default weights
recognizer = MobileFace()  # Lightweight alternative
recognizer = SphereFace()  # Angular softmax alternative
```

### High-Level Detection API

```python
from uniface import detect_faces

# One-line face detection
faces = detect_faces(image, method='retinaface', conf_thresh=0.8)
```

---

## Model Performance

### Face Detection (WIDER FACE Dataset)

| Model              | Easy   | Medium | Hard   | Use Case               |
| ------------------ | ------ | ------ | ------ | ---------------------- |
| retinaface_mnet025 | 88.48% | 87.02% | 80.61% | Mobile/Edge devices    |
| retinaface_mnet_v2 | 91.70% | 91.03% | 86.60% | Balanced (recommended) |
| retinaface_r34     | 94.16% | 93.12% | 88.90% | High accuracy          |
| scrfd_500m         | 90.57% | 88.12% | 68.51% | Real-time applications |
| scrfd_10g          | 95.16% | 93.87% | 83.05% | Best accuracy/speed    |
| yolov5s_face       | 94.33% | 92.61% | 83.15% | Real-time + accuracy   |
| yolov5m_face       | 95.30% | 93.76% | 85.28% | High accuracy          |

_Accuracy values from original papers: [RetinaFace](https://arxiv.org/abs/1905.00641), [SCRFD](https://arxiv.org/abs/2105.04714), [YOLOv5-Face](https://arxiv.org/abs/2105.12931)_

**Benchmark on your hardware:**

```bash
python scripts/run_detection.py --image assets/test.jpg --iterations 100
```

See [MODELS.md](MODELS.md) for detailed model information and selection guide.

<div align="center">
    <img src="assets/test_result.png">
</div>

---

## Examples

### Webcam Face Detection

```python
import cv2
from uniface import RetinaFace
from uniface.visualization import draw_detections

detector = RetinaFace()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    # Extract data for visualization
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]

    draw_detections(frame, bboxes, scores, landmarks, vis_threshold=0.6)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Face Search System

```python
import numpy as np
from uniface import RetinaFace, ArcFace

detector = RetinaFace()
recognizer = ArcFace()

# Build face database
database = {}
for person_id, image_path in person_images.items():
    image = cv2.imread(image_path)
    faces = detector.detect(image)
    if faces:
        embedding = recognizer.get_normalized_embedding(
            image, faces[0]['landmarks']
        )
        database[person_id] = embedding

# Search for a face
query_image = cv2.imread("query.jpg")
query_faces = detector.detect(query_image)
if query_faces:
    query_embedding = recognizer.get_normalized_embedding(
        query_image, query_faces[0]['landmarks']
    )

    # Find best match
    best_match = None
    best_similarity = -1

    for person_id, db_embedding in database.items():
        similarity = np.dot(query_embedding, db_embedding.T)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_id

    print(f"Best match: {best_match} (similarity: {best_similarity:.4f})")
```

More examples in the [examples/](examples/) directory.

---

## Advanced Configuration

### Custom ONNX Runtime Providers

```python
from uniface.onnx_utils import get_available_providers, create_onnx_session

# Check available providers
providers = get_available_providers()
print(f"Available: {providers}")

# Force CPU-only execution
from uniface import RetinaFace
detector = RetinaFace()
# Internally uses create_onnx_session() which auto-selects best provider
```

### Model Download and Caching

Models are automatically downloaded on first use and cached in `~/.uniface/models/`.

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Manually download and verify a model
model_path = verify_model_weights(
    RetinaFaceWeights.MNET_V2,
    root='./custom_models'  # Custom cache directory
)
```

### Logging Configuration

```python
from uniface import Logger
import logging

# Set logging level
Logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR

# Disable logging
Logger.setLevel(logging.CRITICAL)
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uniface --cov-report=html

# Run specific test file
pytest tests/test_retinaface.py -v
```

---

## Development

### Setup Development Environment

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Format code
ruff format .

# Check for linting errors
ruff check .

# Auto-fix linting errors
ruff check . --fix
```

Ruff configuration is in `pyproject.toml`. Key settings:

- Line length: 120
- Python target: 3.10+
- Import sorting: `uniface` as first-party

### Project Structure

```
uniface/
├── uniface/
│   ├── detection/       # Face detection models
│   ├── recognition/     # Face recognition models
│   ├── landmark/        # Landmark detection
│   ├── attribute/       # Age, gender, emotion
│   ├── onnx_utils.py    # ONNX Runtime utilities
│   ├── model_store.py   # Model download & caching
│   └── visualization.py # Drawing utilities
├── tests/               # Unit tests
├── examples/            # Example notebooks
└── scripts/             # Utility scripts
```

---

## References
- **Uniface** [uniface](https://github.com/yakhyo/uniface)
- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) - PyTorch implementation and training code
- **YOLOv5-Face Original**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) - Original PyTorch implementation
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) - ONNX inference implementation
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) - ArcFace, MobileFace, SphereFace training code
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface) - Model architectures and pretrained weights

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/yakhyo/uniface).
