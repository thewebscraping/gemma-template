#!/usr/bin/env python
"""
setup.py - a setup script

Copyright (C) 2024 Tu Pham

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors:
    Tu Pham <thetwofarm@gmail.com>
"""

import os
import re
import sys
from codecs import open

from setuptools import setup

BASE_DIR = os.path.dirname(__file__)
CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 9)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """Python version not supported, you need to use Python version >= {}.{}""".format(
            *REQUIRED_PYTHON
        )
    )
    sys.exit(1)


def normalize(name) -> str:
    name = re.sub(r"\s+", "-", name)
    return re.sub(r"[-_.]+", "-", name).lower()


version = {}
with open(os.path.join(BASE_DIR, "gemma_template", "__version__.py"), "r", "utf-8") as f:
    exec(f.read(), version)

if __name__ == "__main__":
    setup(
        name=version["__title__"],
        version=version["__version__"],
        description=version["__description__"],
        long_description_content_type="text/markdown",
        author=version["__author__"],
        author_email=version["__author_email__"],
        url=version["__url__"],
    )
