from datasets import Dataset, DatasetDict


def assert_dataset_equal(ds, hidden_text: str = None):
    a, b = ds[0], ds[1]
    assert a["is_masked"] is True
    assert b["is_masked"] is False
    assert "_____" in hidden_text
    assert len(ds) == 2


def test_attr(template_instance, data, config):
    attr = template_instance.get_template_attr(**data[0], **config)
    assert "You are a multilingual professional writer." == attr.system_prompt
    assert "_____" in attr.input
    assert "# Role" in attr.instruction
    assert "Custom Title" in attr.prompt_structure
    assert "Custom Title" in attr.output


def test_load_dataset_from_dict(template_instance, data, config):
    text_ds = template_instance.load_dataset(data, output_format='text', **config)

    # Assert
    assert_dataset_equal(text_ds, text_ds[0]["text"])
    assert "Custom Title" in text_ds[0]["text"]
    alpaca_ds = template_instance.load_dataset(data, output_format='alpaca', **config)
    assert_dataset_equal(alpaca_ds, alpaca_ds[0]["input"])
    assert "# Role" in alpaca_ds[0]["instruction"]
    assert "Custom Title" in alpaca_ds[0]["input"]
    assert "Custom Title" in alpaca_ds[0]["output"]
    openai_ds = template_instance.load_dataset(data, output_format='openai', **config)  # noqa
    assert_dataset_equal(openai_ds, openai_ds[0]["messages"][1]["content"])  # noqa
    assert "Custom Title" in openai_ds[0]["messages"][-1]["content"]
    assert "# Role" in openai_ds[0]["messages"][0]["content"]
    assert "Custom Title" in openai_ds[0]["messages"][1]["content"]
    assert "Custom Title" in openai_ds[0]["messages"][2]["content"]


def test_load_dataset_from_Dataset(template_instance, data, config):
    dataset = Dataset.from_list(data)

    # Assert
    text_ds = template_instance.load_dataset(dataset, output_format='text', **config)
    assert_dataset_equal(text_ds, text_ds[0]["text"])
    assert "# Role" in text_ds[0]["text"]
    assert "Custom Title" in text_ds[0]["text"]

    # Assert
    alpaca_ds = template_instance.load_dataset(dataset, output_format='alpaca', **config)
    assert_dataset_equal(alpaca_ds, alpaca_ds[0]["input"])
    assert "# Role" in alpaca_ds[0]["instruction"]
    assert "Custom Title" in alpaca_ds[0]["input"]

    # Assert
    openai_ds = template_instance.load_dataset(dataset, output_format='openai', **config)  # noqa
    assert_dataset_equal(openai_ds, openai_ds[0]["messages"][1]["content"])
    assert "Custom Title" in openai_ds[0]["messages"][-1]["content"]
    assert "# Role" in openai_ds[0]["messages"][0]["content"]
    assert "Custom Title" in openai_ds[0]["messages"][1]["content"]
    assert "Custom Title" in openai_ds[0]["messages"][2]["content"]


def test_load_dataset_from_DatasetDict(template_instance, data, config):
    def split_datadict(ds):
        assert isinstance(ds, DatasetDict)
        return ds["train"], ds["test"]

    dataset = Dataset.from_list(data * 2)
    dataset = dataset.train_test_split(test_size=0.5)

    text_ds = template_instance.load_dataset(dataset, output_format='text', **config)
    train_ds, test_ds = split_datadict(text_ds)

    # Assert Train
    assert_dataset_equal(train_ds, train_ds[0]["text"])
    # Assert Test
    assert_dataset_equal(test_ds, test_ds[0]["text"])

    # Assert Train
    assert_dataset_equal(train_ds, train_ds[0]["text"])
    assert "# Role" in train_ds[0]["text"]
    assert "Custom Title" in train_ds[0]["text"]

    # Assert Test
    assert_dataset_equal(test_ds, test_ds[0]["text"])
    assert "# Role" in test_ds[0]["text"]
    assert "Custom Title" in test_ds[0]["text"]

    alpaca_ds = template_instance.load_dataset(dataset, output_format='alpaca', **config)
    train_ds, test_ds = split_datadict(alpaca_ds)

    # Assert
    assert_dataset_equal(train_ds, train_ds[0]["input"])
    assert_dataset_equal(test_ds, test_ds[0]["input"])

    # Assert Train
    assert "# Role" in train_ds[0]["instruction"]
    assert "Custom Title" in train_ds[0]["input"]

    # Assert Test
    assert "# Role" in test_ds[0]["instruction"]
    assert "Custom Title" in test_ds[0]["input"]

    openai_ds = template_instance.load_dataset(dataset, output_format='openai', **config)
    train_ds, test_ds = split_datadict(openai_ds)

    # Assert
    assert_dataset_equal(train_ds, train_ds[0]["messages"][1]["content"])
    assert_dataset_equal(test_ds, test_ds[0]["messages"][1]["content"])  # noqa

    # Assert Train
    assert "Custom Title" in train_ds[0]["messages"][-1]["content"]
    assert "# Role" in train_ds[0]["messages"][0]["content"]
    assert "Custom Title" in train_ds[0]["messages"][1]["content"]
    assert "Custom Title" in train_ds[0]["messages"][2]["content"]

    # Assert Test
    assert "Custom Title" in test_ds[0]["messages"][-1]["content"]
    assert "# Role" in test_ds[0]["messages"][0]["content"]
    assert "Custom Title" in test_ds[0]["messages"][1]["content"]
    assert "Custom Title" in test_ds[0]["messages"][2]["content"]
