from __future__ import annotations

import re
from collections import Counter

from langdetect import LangDetectException, detect

from .constants import SUPPORTED_LANGUAGES
from .exceptions import LanguageError


def get_n_grams(text: str, n: int = 1):
    """
    Generates n-grams from the input text.

    Args:
        text (str): The input text from which n-grams are to be extracted.
        n (int): The size of the n-grams (default is 1, which corresponds to unigrams).

    Returns:
        list[str]: A list of valid n-grams, filtered to exclude those containing numbers,
                   special characters, or invalid sequences.

    Example:
        >>> get_n_grams("This is a test sentence.", 2)
        ['This is', 'is a', 'a test', 'test sentence']
    """  # noqa: 501

    outputs = []
    for sentence in re.split(r"[\n,.?!:;-]+", text):
        words = re.sub(r"\s+", " ", sentence).split()
        word_n_grams = [words[i : i + n] for i in range(len(words) - n + 1)]
        for items in word_n_grams:
            is_valid = True
            for item in items:
                if re.search(r"[\d\W\-_]", item):
                    is_valid = False
                    break

            if is_valid:
                outputs.append(" ".join(items))
    return outputs


def get_language(text, raise_exception: bool = False):
    """
    Identifies the language of the given input text.

    Args:
        text (str): The input text to analyze for language detection.
        raise_exception (bool): True is raise exception if the language is not supported or cannot be identified,
                                otherwise returns a tuple str: ("unk", "Unknown").

    Returns:
        tuple: A tuple containing:
            - code (str): The language code of the detected language.
            - language (str): The full name of the detected language.

    Raises:
        LanguageError:  If the language is not supported or cannot be identified.

    Example:
        >>> get_language("Bonjour tout le monde")
        ('fr', 'French')
    """  # noqa: E501

    try:
        code = str(detect(text)).lower()
        if code in SUPPORTED_LANGUAGES:
            return code, SUPPORTED_LANGUAGES[code]

    except LangDetectException:
        pass

    except Exception as e:
        print("Unexpected error:", e)

    if raise_exception:
        raise LanguageError("The language is not supported or cannot be identified.")

    return "unk", "Unknown"


def get_frequently_words(
    text: str,
    *,
    n: int = 1,
    response_n: int = 10,
    language_code: str = "auto",
    min_chars_length: int = 2,
    max_chars_length: int = 0,
    excluded_words: list = (),
    raise_exception: bool = False,
) -> list[str]:
    """
    Extracts the most common words (unigrams, bigrams, trigrams) from the input text.

    Args:
        text (str): The input text from which popular words are to be extracted.
        n (int): The size of the n-grams (default is 1 for unigrams).
        response_n (int): The maximum number of results to return (default is 10).
        language_code (str): The language code to filter results by language.
                             Use "auto" to detect language automatically.
        min_chars_length(int): Min chars of word.
        max_chars_length(int): Max chars of word.
        excluded_words(list[str]): List of excluded words.
        raise_exception (bool): True is raise exception if the language is not supported or cannot be identified,
                                otherwise returns a empty list.

    Returns:
        list[str]: A list of the most frequent n-grams matching the specified language.

    Raises:
        LanguageError:  If the language is not supported or cannot be identified.

    Example:
        >>> get_frequently_words("This is a test. This test is simple.", n=1, response_n=3)
        ['test', 'is', 'this']
    """  # noqa: 501

    if str(language_code).lower() not in SUPPORTED_LANGUAGES:
        language_code, _ = get_language(text, raise_exception)

    if language_code == "unk":
        return []

    outputs = []
    excluded_words = (
        excluded_words if isinstance(excluded_words, list) else [excluded_words]
    )
    words = get_n_grams(text, n)
    if words:
        n_valid = 0
        for word, _ in Counter(words).most_common():
            if n_valid == response_n:
                break

            if min_chars_length and len(word) < min_chars_length:
                continue

            if max_chars_length and len(word) > max_chars_length:
                continue

            is_continue = False
            for excluded_word in excluded_words:
                if str(excluded_word) in word:
                    is_continue = True
                    break

            if is_continue:
                continue

            code, _ = get_language(word, raise_exception)
            if code == language_code:
                n_valid += 1
                outputs.append(word)

    return outputs
