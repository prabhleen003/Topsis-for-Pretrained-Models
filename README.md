<h1 align="center">🤖 TOPSIS Analysis for Pre-trained NLP Models</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
  <i>Applying TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) to find the best pre-trained models for various NLP tasks.</i>
</p>

---

## 📌 Problem Statement

**Roll Number:** `102317143`

Apply **TOPSIS method** to evaluate and rank pre-trained models for the following NLP tasks:

| # | Task |
|:-:|------|
| 1 | 📝 Text Summarization |
| 2 | ✍️ Text Generation |
| 3 | 🏷️ Text Classification |
| 4 | 🔗 Text Sentence Similarity |
| 5 | 💬 Text Conversational |
---

## 🎯 What is TOPSIS?

**TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) is a multi-criteria decision-making method that:

1. ✅ Normalizes the decision matrix
2. ✅ Applies weights to criteria
3. ✅ Finds ideal best and ideal worst solutions
4. ✅ Calculates distance from ideal solutions
5. ✅ Ranks alternatives based on relative closeness
```
TOPSIS Score = Distance from Worst / (Distance from Best + Distance from Worst)
```

**Higher TOPSIS Score = Better Model** 🏆

---

## 📂 Tasks & Models Evaluated

### 📝 Task 1: Text Summarization

| # | Model | Description |
|:-:|-------|-------------|
| 1 | t5-small | Google's T5 small version |
| 2 | distilbart-cnn-12-6 | Distilled BART model |
| 3 | bart-large-cnn | Facebook's BART fine-tuned on CNN |
| 4 | pegasus-xsum | Google's PEGASUS for extreme summarization |
| 5 | bart-large-cnn-samsum | BART fine-tuned on SAMSum |

**Criteria:** Inference Time (-), Compression Ratio (+), Quality Score (+)

---

### ✍️ Task 2: Text Generation

| # | Model | Description |
|:-:|-------|-------------|
| 1 | opt-125m | Meta's OPT model |
| 2 | distilgpt2 | Distilled GPT-2 |
| 3 | gpt-neo-125M | EleutherAI's GPT-Neo |
| 4 | gpt2 | OpenAI's GPT-2 base |
| 5 | gpt2-medium | GPT-2 medium version |

**Criteria:** Inference Time (-), Diversity Score (+), Fluency Score (+)

---

### 🏷️ Task 3: Text Classification

| # | Model | Description |
|:-:|-------|-------------|
| 1 | distilbert-base-uncased-finetuned-sst-2-english | DistilBERT for sentiment |
| 2 | bertweet-base-sentiment-analysis | BERTweet for sentiment |
| 3 | sentiment-roberta-large-english | Large RoBERTa sentiment |
| 4 | twitter-roberta-base-sentiment | RoBERTa for Twitter sentiment |
| 5 | bert-base-multilingual-uncased-sentiment | Multilingual BERT |

**Criteria:** Inference Time (-), Avg Confidence (+), Accuracy (+)

---

### 🔗 Task 4: Sentence Similarity

| # | Model | Description |
|:-:|-------|-------------|
| 1 | paraphrase-MiniLM-L6-v2 | MiniLM for paraphrase detection |
| 2 | all-MiniLM-L6-v2 | MiniLM for sentence embeddings |
| 3 | multi-qa-MiniLM-L6-cos-v1 | MiniLM for QA |
| 4 | all-distilroberta-v1 | DistilRoBERTa embeddings |
| 5 | all-mpnet-base-v2 | MPNet base model |

**Criteria:** Inference Time (-), Correlation (+)

---

### 💬 Task 5: Conversational

| # | Model | Description |
|:-:|-------|-------------|
| 1 | DialoGPT-small | Microsoft's DialoGPT small |
| 2 | DialoGPT-medium | DialoGPT medium version |

**Criteria:** Inference Time (-), Diversity (+), Quality (+)

---

## 📊 Results

### 🏆 Best Model for Each Task

| Task | Best Model | TOPSIS Score |
|:-----|:-----------|:------------:|
| 📝 Text Summarization | **t5-small** | 0.65 |
| ✍️ Text Generation | **opt-125m** | 0.80 |
| 🏷️ Text Classification | **distilbert-base-uncased-finetuned-sst-2-english** | 0.95 |
| 🔗 Sentence Similarity | **paraphrase-MiniLM-L6-v2** | 0.65 |
| 💬 Conversational | **DialoGPT-small** | 0.85 |

---

### 📈 Detailed Results

#### 📝 Task 1: Text Summarization

| Rank | Model | TOPSIS Score |
|:----:|:------|:------------:|
| 🥇 1 | t5-small | 0.65 |
| 🥈 2 | distilbart-cnn-12-6 | 0.55 |
| 🥉 3 | bart-large-cnn | 0.45 |
| 4 | pegasus-xsum | 0.35 |
| 5 | bart-large-cnn-samsum | 0.25 |

---

#### ✍️ Task 2: Text Generation

| Rank | Model | TOPSIS Score |
|:----:|:------|:------------:|
| 🥇 1 | opt-125m | 0.80 |
| 🥈 2 | distilgpt2 | 0.65 |
| 🥉 3 | gpt-neo-125M | 0.45 |
| 4 | gpt2 | 0.35 |
| 5 | gpt2-medium | 0.20 |

---

#### 🏷️ Task 3: Text Classification

| Rank | Model | TOPSIS Score |
|:----:|:------|:------------:|
| 🥇 1 | distilbert-base-uncased-finetuned-sst-2-english | 0.95 |
| 🥈 2 | bertweet-base-sentiment-analysis | 0.55 |
| 🥉 3 | sentiment-roberta-large-english | 0.50 |
| 4 | twitter-roberta-base-sentiment | 0.45 |
| 5 | bert-base-multilingual-uncased-sentiment | 0.20 |

---

#### 🔗 Task 4: Sentence Similarity

| Rank | Model | TOPSIS Score |
|:----:|:------|:------------:|
| 🥇 1 | paraphrase-MiniLM-L6-v2 | 0.65 |
| 🥈 2 | all-MiniLM-L6-v2 | 0.55 |
| 🥉 3 | multi-qa-MiniLM-L6-cos-v1 | 0.55 |
| 4 | all-distilroberta-v1 | 0.50 |
| 5 | all-mpnet-base-v2 | 0.20 |

---

#### 💬 Task 5: Conversational

| Rank | Model | TOPSIS Score |
|:----:|:------|:------------:|
| 🥇 1 | DialoGPT-small | 0.85 |
| 🥈 2 | DialoGPT-medium | 0.25 |

---

## 📈 Visualizations

<p align="center">
  <img src="topsis_results.png" width="900" alt="TOPSIS Results"/>
</p>

<p align="center"><i>TOPSIS Rankings for all NLP Tasks - Green bars indicate best performing models</i></p>

---

## 💡 Key Findings
```
✅ Text Classification: distilbert-base-uncased-finetuned-sst-2-english achieved 
   highest TOPSIS score (0.95) among all tasks

✅ Text Generation: opt-125m outperformed larger models like gpt2-medium

✅ Conversational: DialoGPT-small significantly outperformed DialoGPT-medium

✅ Sentence Similarity: paraphrase-MiniLM-L6-v2 showed best balance of speed & accuracy

✅ Text Summarization: t5-small provided best trade-off between quality and speed

✅ Smaller/distilled models often outperform larger models when considering 
   inference time as a factor
```

---

## 🎯 Final Conclusion

| Metric | Result |
|--------|--------|
| 🥇 **Best Overall Task Performance** | Text Classification (0.95) |
| 🏆 **Best Summarization Model** | t5-small |
| 🏆 **Best Generation Model** | opt-125m |
| 🏆 **Best Classification Model** | distilbert-base-uncased-finetuned-sst-2-english |
| 🏆 **Best Similarity Model** | paraphrase-MiniLM-L6-v2 |
| 🏆 **Best Conversational Model** | DialoGPT-small |

### 📌 Key Takeaways

1. **Distilled models win**: Smaller, distilled versions often outperform their larger counterparts when inference time is considered
2. **Task-specific fine-tuning matters**: Models fine-tuned for specific tasks (like SST-2 for sentiment) dominate their categories
3. **TOPSIS provides balanced ranking**: Unlike single-metric comparisons, TOPSIS considers multiple factors for fair evaluation
4. **Speed vs Quality trade-off**: The best models balance inference speed with output quality

---

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TOPSIS_NLP_Assignment.git

# Install dependencies
pip install transformers datasets torch pandas numpy matplotlib seaborn

# Run the notebook
jupyter notebook TOPSIS_NLP_Assignment.ipynb
```

---



## 🛠️ Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

---

## 📚 References

- [TOPSIS Method - Wikipedia](https://en.wikipedia.org/wiki/TOPSIS)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 👤 Author

<h3>Prabhleen Kaur</h3>

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/prabhleen003)

---

<p align="center">
  <b>⭐ Star this repository if you found it helpful!</b>
</p>

---

<p align="center">
  Made with ❤️ using Python & HuggingFace Transformers
</p>
