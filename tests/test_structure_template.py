from gemma_template import StructureField, Template, gemma_template

STRUCTURE_TEMPLATE = """# Response Structure Format:
You must follow the response structure:
{structure_template}

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.
"""  # noqa: E501


def test_structure_template(data_items, config):
    template = gemma_template.template(structure_template=STRUCTURE_TEMPLATE, **data_items[0], **config)
    assert "# Response Structure Format" in template


def test_structure_template_function(data_items, config):
    def structure_fn(
        fn,
        **instruction_kwargs,
    ):
        return "### STRUCTURE TEST"

    template_fn = gemma_template.template(structure_template=structure_fn, **data_items[0], **config)
    assert "### STRUCTURE TEST" in template_fn


def test_fully_custom_structure_template(data_items, config):
    def instruction_fn(
        fn,
        **instruction_kwargs,
    ):
        return "### INSTRUCTION TEST"

    prompt_instance = Template(
        structure_field=StructureField(
            title=["Custom Title"],
            description=["Custom Description"],
            document=["Custom Article"],
            main_points=["Custom Main Points"],
            categories=["Custom Categories"],
            tags=["Custom Tags"],
        ),
    )

    response = prompt_instance.template(
        instruction_template=instruction_fn,
        structure_template=STRUCTURE_TEMPLATE,
        **data_items[0],
        **config
    )

    assert "### INSTRUCTION TEST" in response
    assert "Custom Title" in response
    assert "Custom Description" in response
