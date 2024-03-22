import re
from typing import Iterable


def find_patterns(input_string: str, patterns: str | Iterable[str]) -> list[str]:
    """
    Find all patterns in a piece of text.

    :param input_string: The text to check for JSON objects.
    :param patterns: The raw string pattern(s) to search for with a Regular Expression.
    :param flag: The re flag to use when doing matching.
    :return: list[str] of strings found.
    """
    # If only a string is passed in then create a list with only
    # that string in it
    if isinstance(patterns, str):
        patterns = [patterns]

    results = []
    for pattern in patterns:
        found = re.findall(pattern, input_string, re.DOTALL)
        found = [f.strip() for f in found]
        results.extend(found)

    return results
