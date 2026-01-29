# Abstract Generator Using Transformer

A machine learning project that implements an abstractive text summarization system using Transformer architecture. This project processes research paper abstracts from the CORD-19 dataset and prepares them for training a neural network model capable of generating summaries.

## Project Overview

This Jupyter notebook demonstrates a complete pipeline for building an abstract generation model:

- **Data Loading & Exploration**: Load and analyze research paper metadata from the CORD-19 dataset
- **Data Cleaning & Preprocessing**: Clean text data, handle missing values, and normalize text
- **Embedding Preparation**: Tokenize text and prepare embeddings for neural network training
- **Model Architecture**: Implement transformer-based neural network for abstract generation

## Features

- **Dataset**: CORD-19 Research Challenge dataset (100-sample subset for demonstration)
- **Text Processing**:
  - Punctuation and special character removal
  - Text normalization (lowercase conversion)
  - Tokenization with vocabulary of 10,000 most frequent words
  - Sequence padding to uniform length (100 tokens)
- **Neural Network Components**:
  - Embedding layers for word representation
  - Transformer architecture for sequence-to-sequence learning
  - Dropout for regularization
  - Dense layers for output generation

## Prerequisites

- Python 3.7+
- TensorFlow/Keras
- Pandas
- NumPy
- Access to CORD-19 dataset

## Installation

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Dataset

The project uses the CORD-19 dataset from Kaggle. Columns used:

- `cord_uid`: Unique identifier for the research paper
- `title`: Title of the research paper
- `abstract`: Abstract of the research paper

## Project Structure

1. **Load and Explore the Dataset**: Load data from CSV and examine structure
2. **Data Cleaning and Preprocessing**: Remove irrelevant columns, handle missing values, clean text
3. **Embedding Preparation**: Tokenize texts and create padded sequences for model input
4. **Output Readiness Check**: Verify shape consistency and data integrity

## Usage

Run the Jupyter notebook cell by cell to execute the pipeline:

```bash
jupyter notebook abstract-generator-using-transformer.ipynb
```

## Model Architecture

The model follows a transformer-based sequence-to-sequence architecture:

- Input layer: Padded token sequences (batch_size × 100)
- Embedding layer: Converts tokens to dense vectors
- Transformer blocks: Multi-head attention and feed-forward layers
- Output layer: Generates abstract/summary tokens

## Results

The notebook generates:

- Cleaned and tokenized dataset ready for training
- Padded sequences suitable for batch processing
- Model-ready data format with consistent shapes

## Future Enhancements

- Train the complete transformer model on larger dataset
- Implement BLEU/ROUGE evaluation metrics
- Add beam search for summary generation
- Deploy as API endpoint

## License

This project uses the CORD-19 dataset. Please refer to the CORD-19 dataset license for usage terms.

## Author

Created as a demonstration of transformer-based text summarization techniques.

## References

- [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
- [TensorFlow Transformers](https://www.tensorflow.org/text/tutorials/transformer)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
