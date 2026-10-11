# Speech Emotion Recognition for Tunisian Dialect

![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-wav2vec_2.0-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?logo=huggingface&logoColor=black)

Given a short audio clip of Tunisian dialect, predict whether the speaker sounds **angry, happy, neutral or sad**.

This is my end-of-study project, done during an internship at iCompass. The goal was a model that could sit inside a **voicebot**, so that a bot answering a phone call can react to how the caller sounds and not only to what the transcript says.

---

## Why this project exists

Most emotion datasets and models are built for English, and Arabic resources mostly cover Modern Standard Arabic. Tunisian dialect has very little, so I had to build the data myself before any modelling.

- The clips come from a mix of sources: company data from iCompass, and audio from Tunisian TV series, talk shows and podcasts. I converted the videos to audio and cut them into segments with a single speaker each.
- Every segment was labelled by hand with one of four emotions, following an annotation guideline.
- The corpus is imbalanced (`neutral` and `sad` dominate, `happy` is the smallest class), so I added augmentation to the training set: white noise, time stretching and pitch shifting.

Then I compared three approaches on the same test clips: classical machine learning on hand-crafted features, LSTM networks, and fine-tuning **wav2vec 2.0 XLSR**.

---

## Results

All models are evaluated on the same **1,025 test clips** (228 angry, 108 happy, 394 neutral, 295 sad). The classes are unbalanced, so **macro F1** (the average of the four per-class F1 scores) is the main metric. A model that always answered `neutral` would reach 38% accuracy.

| Approach | Best model | Features | Accuracy | Macro F1 |
|---|---|---|---|---|
| Classical ML | Random forest (100 trees) | MFCC + chroma + mel (180 values) | 0.64 | 0.55 |
| Deep learning | 3 stacked LSTM layers + batch norm | MFCC (40 values) | 0.63 | 0.58 |
| **wav2vec 2.0 XLSR (fine-tuned)** | `wav2vec2-large-xlsr-53-arabic` + classification head | Raw waveform | **0.69** | **0.68** |

Fine-tuning wav2vec 2.0 gives the best result by a clear margin, about 0.10 macro F1 above the best LSTM.

**wav2vec 2.0, per class**

| Emotion | Precision | Recall | F1 | Test clips |
|---|---|---|---|---|
| angry | 0.56 | 0.84 | 0.67 | 228 |
| happy | 0.60 | 0.62 | 0.61 | 108 |
| neutral | 0.79 | 0.54 | 0.64 | 394 |
| sad | 0.78 | 0.81 | 0.79 | 295 |

![Per-class metrics](results/wav2vec2_per_class_metrics.png)

**What I take from these numbers**

- Sad is the easiest emotion (F1 0.79). Happy is the hardest (F1 0.61), and it is also the smallest class.
- The model likes to say "angry": recall is 0.84 but precision only 0.56. It misses almost half of the neutral clips (recall 0.54).
- Training loss goes from 0.81 down to 0.27 while validation loss stays around 0.95. The model overfits, even though validation accuracy still climbs from 61% to 70% over four epochs.

![Training curves](results/wav2vec2_training_curves.png)

---

## The data

- **Size:** 4,906 clips, split 80/20 into 3,881 training and 1,025 test clips.
- **Augmentation:** applied to the training split only, after the split. It brings the training set from 3,881 to 5,587 clips. The test set is the same 1,025 clips in every experiment, so the models are directly comparable.
- **Not included:** the audio. It comes from TV and podcast recordings plus company data that I cannot redistribute.

---

## How the wav2vec 2.0 model is built

- **Base model:** [`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic), an XLSR-53 model already fine-tuned on Arabic speech.
- **Architecture:** wav2vec 2.0 encoder, mean pooling over time, then a new classification head (dropout, dense, tanh, dropout, 4-way output) that starts from random weights.
- **What is trained:** the convolutional feature extractor is frozen. The transformer layers and the new head are trained.
- **Training:** 4 epochs (2,792 steps), batch size 4 with gradient accumulation 2 (effective batch 8), learning rate 1e-4, cross-entropy loss, about 97 minutes on a Colab GPU.

---

## Notebooks

| Notebook | What it does |
|---|---|
| [`01_data_augmentation.ipynb`](notebooks/01_data_augmentation.ipynb) | Noise, time-stretch and pitch-shift augmentation (runs on a synthetic demo signal) |
| [`02_classical_ml_features.ipynb`](notebooks/02_classical_ml_features.ipynb) | MFCC, chroma and mel features, SVM / decision tree / random forest, per-class results |
| [`03_lstm_experiments.ipynb`](notebooks/03_lstm_experiments.ipynb) | LSTM on each feature set, imbalanced vs augmented training, deeper architecture |
| [`04_wav2vec2_xlsr_finetuning.ipynb`](notebooks/04_wav2vec2_xlsr_finetuning.ipynb) | Fine-tuning wav2vec 2.0 XLSR for 4-class emotion recognition |

The saved outputs (training logs, classification reports) are those of my **original runs on Google Colab**. I cleaned and documented the code afterwards, but I could not re-run it on the original audio. Notebooks 01 to 03 were run on synthetic audio to check that the code works, and notebook 04 needs a GPU and was not re-run. Two parts are not original code, and both are marked in the notebooks: the augmentation parameters (never recorded) and the BiLSTM cell (rebuilt from the architecture diagram in my thesis).

---

## How to use it

Install the dependencies:

```bash
git clone https://github.com/wissalzoghbi/tunisian-speech-emotion-recognition
cd tunisian-speech-emotion-recognition
pip install -r requirements.txt
```

**Notebooks 01 to 03** (augmentation, classical ML, LSTM). Arrange your own labelled audio like this:

```
data/
├── annotations.xlsx             # columns: path (e.g. 15-3.wav), label (angry / happy / neutral / sad)
├── train/<recording>/<recording>-<n>.wav
└── val/<recording>/<recording>-<n>.wav     # used as the test set
```

Edit the configuration cell (`DATA_DIR`, `FEATURES`) and run all cells. Notebook 01 runs without data. The LSTM notebook needs TensorFlow 2.16 or later.

**Notebook 04** (wav2vec 2.0), on Google Colab with a GPU runtime:

1. Prepare two tab-separated files, `train.csv` and `test.csv`, with the columns `name`, `path`, `emotion`.
2. Edit the paths in the configuration cell and run all cells.

---

## Limitations

These points matter when you read the numbers above.

- **The same recording can appear in both train and test.** I split each recording folder clip by clip, not by speaker or by recording. A model can partly learn a speaker's voice or the recording conditions, so the scores are probably higher than what new speakers would get.
- **The test set was also used for model selection.** It served as the validation set during training and to pick hyperparameters, so the final scores are slightly optimistic.
- **One run per model.** There is no cross-validation or confidence interval, so differences of a few points between the classical models and the LSTM are within what another random seed could change.
- **Mixed training data, single-source test data.** Training mixes company data, YouTube audio and augmented copies, while the test set only has company data. How well the models transfer to the other sources has not been measured.
- **Acted and TV speech, not phone calls.** I have not evaluated phone-quality audio or unseen speakers, which is the real voicebot setting.
- **The LSTM does not read time.** Each clip is first averaged into one vector of 40 MFCC values, and the LSTM reads that vector as a sequence. The recurrence runs over the feature dimension, not over time.
- **Pooling in the wav2vec 2.0 model also averages padded frames**, which matters when clips have different lengths.
- **Some original material was lost.** The augmentation parameters, the original BiLSTM code and some feature-by-feature runs were not recovered. No convolutional network was trained.

---

## Credits

- Thesis: W. Zoghbi, *Implementation of Speech Emotion Recognition System for Tunisian Arabic Dialect*, graduation project, ESSAI, University of Carthage. The three augmentation functions and the BiLSTM architecture come from it.
- The wav2vec 2.0 model code and training loop are adapted from the Greek speech emotion recognition notebook in [m3hrdadfi/soxan](https://github.com/m3hrdadfi/soxan).
- The hand-crafted feature extraction follows [x4nth055/emotion-recognition-using-speech](https://github.com/x4nth055/emotion-recognition-using-speech).
- Pretrained model: [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic), built on [wav2vec 2.0 / XLSR-53](https://arxiv.org/abs/2006.13979).

## Author

**Wissal Zoghbi**, data science and AI engineer working on Arabic speech and NLP.
[LinkedIn](https://www.linkedin.com/in/wissal-zoghbi) · [GitHub](https://github.com/wissalzoghbi)
