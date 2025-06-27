# **Speech Emotion Recognition using Wav2Vec2.0**

## **Abstract**

This project aims to develop a robust Speech Emotion Recognition (SER) system using the RAVDESS dataset and Wav2Vec2.0 — a state-of-the-art self-supervised speech representation model by Facebook AI. The objective is to accurately classify human emotions directly from raw speech audio, leveraging the power of transformer-based models. Traditional feature extraction methods are compared briefly, but the core focus is on fine-tuning Wav2Vec2.0 for emotion classification, enabling real-time and production-ready performance.



## **Introduction**

Emotion is fundamental to human communication. Detecting emotion from speech enables machines to interact more naturally and empathetically with users. This task is complex due to the subtleties in vocal cues, tone, pitch, and rhythm. Traditional machine learning models using hand-crafted features often fall short in capturing these nuances. Recently, deep learning models like CNNs and LSTMs have improved performance, but Wav2Vec2.0 offers a new paradigm—processing raw waveform directly using self-supervised learning, significantly simplifying preprocessing and boosting accuracy.



## **Dataset**

The project uses the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), which includes 1,440 audio files from 24 actors expressing 8 emotions:

* Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

The dataset was downloaded from KaggleHub and preprocessed with `torchaudio`.


## **Methodology**

### 1. **Feature Extraction (Traditional Baseline - Optional)**

Initial experiments used MFCCs, chroma, tonnetz, and spectral contrast via Librosa for traditional ML models (e.g., Random Forest, Logistic Regression). However, they were slow to scale and underperformed compared to deep learning alternatives.

### 2. **Wav2Vec2.0-based Emotion Classifier**

* **Pretrained Backbone**: `facebook/wav2vec2-base`
* **Input**: Raw `.wav` files sampled at 16 kHz
* **Architecture**:

  * Extract last hidden state from Wav2Vec2
  * Use the first token (CLS-like) for classification
  * Custom classifier head:

    * Linear → ReLU → Dropout → Linear → Softmax (8 outputs)

### 3. **Model Training**

* Optimizer: AdamW
* Loss: CrossEntropyLoss
* Dataset split: 80% Train / 20% Test
* Training on GPU using PyTorch



## **Tools and Libraries**

* Python (Jupyter Notebook)
* PyTorch & Torchaudio
* Hugging Face Transformers
* Librosa (for optional traditional features)
* Pandas, NumPy, Matplotlib
* RAVDESS Dataset via KaggleHub



## **Results**

* **Accuracy**: Achieved approximately **87–90% accuracy** on the test set using Wav2Vec2.0.
* **Confusion Matrix**: Most confusion occurred between similar emotions like "calm" vs. "neutral" and "sad" vs. "fearful."
* **Traditional Models**: Scored around 65–72% accuracy with handcrafted MFCC + classical ML pipelines.
* **Wav2Vec2.0 fine-tuning** clearly outperformed all other approaches in both accuracy and training time.


## **Conclusion**

This project demonstrates the effectiveness of using Wav2Vec2.0 for speech emotion recognition. Compared to traditional ML approaches, it requires less manual feature engineering and offers higher accuracy. The system generalizes well across different actors and could be used in real-world applications like customer service bots, mental health monitoring, and sentiment-aware assistants. Future improvements could include training on multilingual datasets, incorporating gender/age variation, or experimenting with Wav2Vec2-Large for further gains.

