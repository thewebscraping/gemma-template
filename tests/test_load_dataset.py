from datasets import Dataset

from gemma_template import gemma_template


def assert_dataset_equal(ds, input_field: str = 'text', output_field: str = 'text'):
    a, b = ds[0], ds[1]
    assert input_field in a
    assert output_field in a
    assert input_field in b
    assert output_field in b
    assert a["is_masked"] is True
    assert b["is_masked"] is False
    assert "_____" in a[input_field]
    assert len(ds) == 2


def test_load_dataset_from_dict(data_items, config):
    text_ds = gemma_template.load_dataset(data_items, output_format='text', **config)
    assert_dataset_equal(text_ds, "text", "text")
    alpaca_ds = gemma_template.load_dataset(data_items, output_format='alpaca', **config)
    assert_dataset_equal(alpaca_ds, "input", "output")
    gpt_ds = gemma_template.load_dataset(data_items, output_format='gpt', **config)
    assert_dataset_equal(gpt_ds, "human", "gpt")


def test_load_dataset_from_Dataset(data_items, config):
    dataset = Dataset.from_list(data_items)
    text_ds = gemma_template.load_dataset(dataset, output_format='text', **config)
    assert_dataset_equal(text_ds, "text", "text")
    alpaca_ds = gemma_template.load_dataset(dataset, output_format='alpaca', **config)
    assert_dataset_equal(alpaca_ds, "input", "output")
    gpt_ds = gemma_template.load_dataset(dataset, output_format='gpt', **config)
    assert_dataset_equal(gpt_ds, "human", "gpt")


def test_load_dataset_from_DatasetDict(data_items, config):
    dataset = Dataset.from_list(data_items * 2)
    dataset = dataset.train_test_split(test_size=0.5)
    text_ds = gemma_template.load_dataset(dataset, output_format='text', **config)
    assert_dataset_equal(text_ds["train"], "text", "text")
    assert_dataset_equal(text_ds["test"], "text", "text")
    alpaca_ds = gemma_template.load_dataset(dataset, output_format='alpaca', **config)
    assert_dataset_equal(alpaca_ds["train"], "input", "output")
    assert_dataset_equal(alpaca_ds["test"], "input", "output")
    gpt_ds = gemma_template.load_dataset(dataset, output_format='gpt', **config)
    assert_dataset_equal(gpt_ds["train"], "human", "gpt")
    assert_dataset_equal(gpt_ds["test"], "human", "gpt")
