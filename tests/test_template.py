

def test_template(template_instance, data, config):
    template = template_instance.apply_template(**data[0], **config)
    assert "_____" in template
    assert "# Role" in template
    assert "# Response Structure Format:" in template
    assert "<start_of_turn>user" in template
    assert "<start_of_turn>model" in template
