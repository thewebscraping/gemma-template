# Quickstart Guide for Gemma Template

**Installation**
----------------

To install the library, you can choose between two methods:

#### **1\. Install via PyPI:**

```shell
pip install gemma-template
```

#### **2\. Install via GitHub Repository:**

```shell
pip install git+https://github.com/thewebscraping/gemma-template.git
```

**Quick Start**
----------------
Start using Gemma Template with just a few lines of code:

## Load Dataset
Returns: A Hugging Face Dataset or DatasetDict object containing the processed prompts.

**Load Dataset from data dict**
```python
prompt_instance = Template()
data_dict = [
    {
        "id": "JnZJolR76_u2",
        "title": "Sample title",
        "description": "Sample description",
        "document": "Sample document",
        "categories": ["Topic 1", "Topic 2"],
        "tags": ["Tag 1", "Tag 2"],
        "output": "Sample output",
        "main_points": ["Main point 1", "Main point 2"],
    }
]
dataset = prompt_instance.load_dataset(data_dict, output_format='text')   # enum: `text`, `alpaca` and `openai`.
print(dataset['text'][0])
```

**Load Dataset from local file path or HuggingFace dataset**
```python
dataset = gemma_template.load_dataset(
    "YOUR_JSON_FILE_PATH_OR_HUGGINGFACE_DATASET",
    # enum: `text`, `alpaca` and `openai`.
    output_format='text',
    # Percentage of documents that need to be word masked.
    # Min: 0, Max: 1. Default: 0.
    max_hidden_ratio=.1,
    # Replace 10% of words in the input document with '_____'.
    # Use int to extract the correct number of words. The `max_hidden_ratio` parameter must be greater than 0.
    max_hidden_words=.1,
    # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    min_chars_length=2,
    # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
    max_chars_length=8,
)
```
