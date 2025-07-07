# Language Model Comparison on the IMDB Dataset

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)

A comprehensive, hands-on comparison of N-gram, from-scratch LSTM, and fine-tuned GPT-2 language models on the IMDB movie review dataset. This project implements each model, trains it on the data, and evaluates it on perplexity and text generation quality.

The entire analysis is contained within the `Analysis.ipynb` notebook.

## Models Compared

This repository explores three distinct language modeling paradigms:

1.  **Statistical N-gram Model:** A classic Trigram (n=3) model using Add-1 (Laplace) smoothing to handle data sparsity. It serves as a strong statistical baseline.
2.  **Recurrent Neural Network (LSTM):** An LSTM-based neural language model built from scratch in PyTorch. This demonstrates the capabilities of recurrent architectures for sequence modeling without relying on pre-training.
3.  **Fine-tuned Transformer (GPT-2):** The pre-trained "gpt2" (124M parameters) model from Hugging Face, fine-tuned on the IMDB dataset. This showcases the power of transfer learning with large language models.

## Key Results

The models were evaluated on their ability to assign probabilities to sentences (measured by perplexity, where lower is better) and to generate coherent text.

### Perplexity Evaluation

Perplexity measures how "surprised" a model is by a sequence of text. A lower score indicates the model assigned a higher probability to the text, suggesting a better fit.

| Model                     | Perplexity (Correct Sentences) | Perplexity (Incorrect Sentences) |
| :------------------------ | :----------------------------: | :------------------------------: |
| **N-gram (Trigram)**      |            5568.18             |            50105.90            |
| **LSTM (from scratch)**   |             134.98             |             544.18             |
| **GPT-2 (Fine-tuned)**    |              13.28             |             308.82             |

***Conclusion:*** The fine-tuned GPT-2 model achieves a dramatically lower perplexity, demonstrating its superior understanding of English grammar and context. The LSTM significantly outperforms the N-gram model, but cannot compete with the knowledge baked into the pre-trained transformer.

### Text Generation Samples

Each model was given the same prompts to generate a text continuation.

| Prompt        | N-gram (Greedy)                                                                                                    | LSTM (Greedy)                    | GPT-2 (Sampled)                                                                                                                                     |
| :------------ | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| I think       | `I think 'm the the one most the part film the is best the of film the is best the of film the is best...`            | `I think it is a`                | `I think you have your answer. We had one scene in this movie. It was the last scene. The movie is very cheesy and pretentious. We don'`             |
| She goes to   | `She goes to the be best described the the one most the part film the is best the of film the is best...`             | `She goes to the movie . the movie is a` | `She goes to bed after a while, he is sleeping and wakes up to a note from his brother saying he was found dead and his father. The young couple has` |
| The movie was | `The movie was the in movie the is best the of film the is best the of film the is best the of film...`               | `The movie was a`                | `The movie was terrible and almost as bad as "Manhunt" or "A Day in the Life of the Lambs" and the movie's main characters are so`                  |
| It felt like  | `It felt like the . film the is best the of film the is best the of film the is best the of film the is...`            | `It felt like the movie . i was a`     | `It felt like a remake of the popular movie "The Big Short" with the idea of the director going from a young boy to a serious and committed man. But` |
| Blue dog      | `Blue dog the , film and . the the one most the part film the is best the of film the is best the of...`               | `Blue dog is a`                  | `Blue dog's name and its dog's name are mentioned throughout the entire movie, but they're never really mentioned. It's very well-written, though I'`   |

***Conclusion:*** The N-gram model quickly falls into repetitive loops. The LSTM produces grammatically simple but coherent starts. The fine-tuned GPT-2 generates far more complex, creative, and contextually relevant text.

## Setup and Usage

To run this analysis yourself, follow these steps.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/Language-Model-Comparison-IMDB.git
cd Language-Model-Comparison-IMDB
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install dependencies:**

A `requirements.txt` file is not included, but you can install all necessary packages with the following command:
```bash
pip install pandas numpy nltk torch transformers datasets accelerate
```
The notebook was tested with `torch>=2.1.0` and `transformers>=4.35.0`.

**4. Download NLTK data:**
The first time you run the notebook, it will download the necessary NLTK packages (`punkt`, `wordnet`, etc.).

**5. Download the Dataset:**
*   Download the "IMDB Dataset of 50K Movie Reviews" from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
*   Place the `IMDB Dataset.csv` file in the root of the project directory.
*   **Rename the file to `imdb.csv`** so the notebook can find it.

**6. Run the Jupyter Notebook:**
```bash
jupyter notebook "Analysis.ipynb"
```
> **Note:** Training the LSTM and fine-tuning GPT-2 are computationally intensive and can take several hours on a CPU. The notebook uses subsets of the data to make this feasible, but a GPU is highly recommended for faster execution.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
