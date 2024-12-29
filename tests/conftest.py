import pytest  # noqa


@pytest.fixture
def data_items():
    return [
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


@pytest.fixture
def config():
    return dict(
        max_hidden_ratio=.5,
        max_hidden_words=.1,
        min_chars_length=2,
        max_chars_length=8,
    )
