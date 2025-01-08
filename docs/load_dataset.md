# Load Dataset

Processes and loads a dataset, generating prompts based on the provided templates.
This function supports various input formats such as file paths, dictionaries, or Hugging Face Dataset objects.
It uses templates to create structured prompts and supports concurrent processing for efficiency.

## Parameters

***

## Sample data
```python
data_dict = [
    {
        "id": "JnZJolR76_u2",
        "title": "Gemma open models",
        "description": "Gemma: Introducing new state-of-the-art open models",
        "document": "Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
        "categories": ["Topic 1", "Topic 2"],
        "tags": ["Tag 1", "Tag 2"],
        "output": "Sample output",
        "main_points": ["Main point 1", "Main point 2"],
    },
    {
        "id": "JnZJolR76_u2",
        "title": "Gemma open models",
        "description": "Gemma: Introducing new state-of-the-art open models",
        "document": "Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
        "categories": ["Topic 1", "Topic 2"],
        "tags": ["Tag 1", "Tag 2"],
        "output": "Sample output",
        "main_points": ["Main point 1", "Main point 2"],
    },
]
```

***

## Text Dataset
Generate Text dataset Format

```pycon
>>> from gemma_template import gemma_template
>>> dataset = gemma_template.load_dataset(data_dict, output_format='text')
>>> dataset
Dataset({
    features: ['text', 'analysis', 'is_masked', 'origin_data'],
    num_rows: 2
})
>>> dataset[0]
{
    'text': '<start_of_turn>user\nYou are...<end_of_turn>\n<start_of_turn>model\n## **Title:**...<end_of_turn>\n',
    'analysis': {'bigrams': ['technology as'],
    'keyword_value': 'Tag 1, Tag 2',
    'language': 'English',
    'language_code': 'en',
    'topic_value': 'Topic 1, Topic 2',
    'trigrams': ['technology as Gemini'],
    'unigrams': ['and', 'built', 'from', 'the', 'research']},
    'is_masked': False,
    'origin_data': {}
}
```

***

## Alpaca Dataset
Generate Alpaca dataset format.

```pycon
>>> from gemma_template import gemma_template
>>> dataset = gemma_template.load_dataset(data_dict, output_format='alpaca')
>>> dataset
Dataset({
    features: ['instruction', 'input', 'output', 'analysis', 'is_masked', 'origin_data'],
    num_rows: 2
})
>>> dataset[0]
{
    'instruction': 'You are a multilingual professional writer...',
    'input': '# Input Text:\nRewrite the input text..',
    'output': '## **Title:**\n### Gemma open models\n\n## **Meta Description:**\nGemma: Introducing new state-of-the-art open models...',
    'analysis': {'bigrams': ['technology as'],
    'keyword_value': 'Tag 1, Tag 2',
    'language': 'English',
    'language_code': 'en',
    'topic_value': 'Topic 1, Topic 2',
    'trigrams': ['technology as Gemini'],
    'unigrams': ['and', 'built', 'from', 'the', 'research']},
    'is_masked': False,
    'origin_data': {}
}
```

***

## OpenAI Dataset
Generate OpenAI dataset format.

```pycon
>>> from gemma_template import gemma_template
>>> dataset = gemma_template.load_dataset(data_dict, output_format='openai')
>>> dataset
Dataset({
    features: ['messages', 'analysis', 'is_masked', 'origin_data'],
    num_rows: 2
})
>>> dataset[0]
{
    'messages': [
        {
            'content': 'You are a multilingual professional writer...',
            'role': 'developer'
        },
        {
            'content': '# Input Text:\nRewrite the input text...',
            'role': 'user'
        },
        {
            'content': '## **Title:**\n### Gemma open models...',
            'role': 'assistant'
        }
    ],
    'analysis': {
        'bigrams': ['technology as'],
        'keyword_value': 'Tag 1, Tag 2',
        'language': 'English',
        'language_code': 'en',
        'topic_value': 'Topic 1, Topic 2',
        'trigrams': ['technology as Gemini'],
        'unigrams': ['and', 'built', 'from', 'the', 'research']
    },
    'is_masked': False,
    'origin_data': {}
}
```
