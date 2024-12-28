from __future__ import annotations

from typing import Callable, Literal, Optional, Union

TemplateTypes = Union[str, Callable]
ExcludedFieldTypes = Optional[
    list[
        Union[
            str,
            Literal[
                "title", "description", "document", "main_points", "categories", "tags"
            ],
        ]
    ]
]
