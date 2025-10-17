# Week 3 Assignment - AI For Software Engineering

This repository contains my solutions for Week 3 of the AI/FORSE program, demonstrating comprehensive machine learning skills across classical ML, deep learning, and natural language processing.

## 📁 Project Structure

```
Week3_Assignment/
├── 📊 Wk3.pdf                   # Complete assignment report and documentation
├── ⚙️ requirements.txt           # Python dependencies
├── 🌸 task1.py                  # Classical ML - Iris Classification
├── 🔢 task2.py                  # Deep Learning - MNIST Digit Recognition  
├── 💬 task3.py                  # NLP - Amazon Reviews Analysis
```

## 🚀 Files Overview

### **📊 Wk3.pdf** 
**Complete Assignment Report**
- Theoretical understanding of ML frameworks (TensorFlow vs PyTorch, Scikit-learn, spaCy)
- Detailed analysis of all three practical tasks
- Performance metrics and visualizations
- Ethical considerations and bias analysis
- Professional documentation of methodologies and results

### **🌺 task1.py - Classical Machine Learning**
**Iris Species Classification using Scikit-learn**
- **Objective**: Classify iris flower species using Decision Trees
- **Dataset**: Iris Species Dataset (built-in)
- **Features**: 
  - Data preprocessing and exploration
  - Decision Tree Classifier implementation
  - Model evaluation (accuracy, precision, recall)
  - Feature importance analysis
- **Results**: High-accuracy species classification with interpretable model

### **🔢 task2.py - Deep Learning** 
**MNIST Handwritten Digit Recognition using TensorFlow**
- **Objective**: Build CNN to classify handwritten digits (0-9)
- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Features**:
  - Convolutional Neural Network architecture
  - Batch normalization and dropout for regularization
  - Optimized training with callbacks
  - Comprehensive evaluation and visualization
- **Results**: **99.42% test accuracy** - exceeds 95% requirement
- **Key Achievement**: Near-perfect digit recognition with minimal misclassifications

### **💬 task3.py - Natural Language Processing**
**Amazon Reviews Analysis using spaCy**
- **Objective**: Perform NER and sentiment analysis on product reviews
- **Dataset**: Sample Amazon product reviews
- **Features**:
  - Named Entity Recognition for brand and product extraction
  - Rule-based sentiment analysis (positive/negative)
  - Entity and sentiment visualization
  - Business insights generation
- **Results**: Successful entity extraction and sentiment patterns identification

### **⚙️ requirements.txt**
**Python Dependencies**
```
tensorflow>=2.13.0
numpy>=1.24.3
Pillow>=10.0.0
matplotlib>=3.7.2
scikit-learn>=1.3.0
spacy>=3.7.0
pandas>=2.0.0
seaborn>=0.12.0
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/fredxotic/Week-3.git
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Download spaCy English model
   python -m spacy download en_core_web_sm
   ```

4. **Run individual tasks**
   ```bash
   # Task 1 - Iris Classification
   python task1.py
   
   # Task 2 - MNIST Digit Recognition
   python task2.py
   
   # Task 3 - NLP Analysis
   python task3.py
   ```

## 📊 Key Results & Achievements

### **Model Performance**
- **MNIST CNN**: 99.42% test accuracy (65 misclassifications out of 10,000)
- **Iris Classifier**: High accuracy with interpretable decision boundaries
- **NLP Pipeline**: Effective entity recognition and sentiment analysis

### **Technical Skills Demonstrated**
- ✅ Classical ML with Scikit-learn
- ✅ Deep Learning with TensorFlow/Keras
- ✅ NLP with spaCy
- ✅ Model evaluation and visualization
- ✅ Ethical AI considerations
- ✅ Professional documentation

## 👨‍💻 Author

**Fred Kaloki**  
AI For Software Engineering  
