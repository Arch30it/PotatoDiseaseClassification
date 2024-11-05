# Potato Disease Classification 🥔

## Overview
This project implements a deep learning solution for classifying diseases in potato plants using leaf images. The system can identify three different conditions:
- Healthy leaves
- Early Blight
- Late Blight

## Features
- Real-time disease prediction from leaf images
- Interactive web interface built with Streamlit
- Pre-trained CNN model using TensorFlow
- High accuracy disease classification
- Detailed recommendations for disease management

## Tech Stack
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- PIL (Python Imaging Library)
- NumPy

## Dataset
The project uses the PlantVillage dataset, which contains images of potato plant leaves. The dataset is organized into three categories:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

## Training

The training script includes:
- Data augmentation
- Model architecture definition
- Training configuration
- Performance visualization

## Usage
1. Open the Streamlit app in your browser
2. Upload an image of a potato plant leaf
3. The app will display:
   - Disease prediction
   - Confidence score
   - Treatment recommendations

## Requirements
```
tensorflow>=2.8.0
streamlit>=1.8.0
pillow>=8.0.0
numpy>=1.19.0
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
