"""Text normalisation, and the unit bug that withheld correct answers.

The numeric check in the grounding validator is the cheapest effective
hallucination trap in a domain full of thresholds. It is also the check most
likely to do damage when it is wrong, because a false positive suppresses a
*correct* answer. These tests pin the behaviour down.
"""

from __future__ import annotations

import pytest

from foundry.text import content_terms, numbers_with_units, sentences, stem


@pytest.mark.parametrize(
    "written,symbol",
    [
        ("limited to a rating of 100 watt hours (Wh) per battery", "limited to 100 Wh"),
        ("2 grams of lithium content", "limited to 2 g of lithium"),
        ("onset at 130 degrees Celsius", "onset at 130 °C"),
        ("a gap of at least 0.7m between rows", "a gap of at least 0.7 metres"),
        ("rated 5 Ah", "rated 5 amp hours"),
        ("a 250 kW charger", "a 250 kilowatt charger"),
    ],
)
def test_a_spelled_out_unit_matches_its_symbol(written, symbol):
    """Regression: "100 watt hours" once yielded "100w" and "100 Wh" yielded
    "100wh", so a correct answer read as a fabricated number and was withheld."""
    assert numbers_with_units(written) & numbers_with_units(symbol)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("at least 10 fatalities occurred", {"10"}),
        ("records retained for 5 years", {"5"}),
        ("inspected every 180 days", {"180"}),
    ],
)
def test_a_trailing_noun_is_not_mistaken_for_a_unit(text, expected):
    assert numbers_with_units(text) == expected


def test_different_magnitudes_still_differ():
    """The trap must keep catching what it is for."""
    assert not (numbers_with_units("exceeds 80 °C") & numbers_with_units("exceeds 65 °C"))
    assert not (numbers_with_units("100 Wh") & numbers_with_units("275 Wh"))


def test_integer_and_decimal_forms_of_the_same_value_agree():
    assert numbers_with_units("80 °C") & numbers_with_units("80.0 °C")


def test_thousands_separators_are_normalised():
    assert numbers_with_units("1,200 Wh") & numbers_with_units("1200 Wh")


def test_ranges_are_extracted_as_both_endpoints():
    found = numbers_with_units("exceeding 101-160 Wh per battery")
    assert "160wh" in found and "101" in found


@pytest.mark.parametrize(
    "a,b",
    [("gas", "gases"), ("vent", "venting"), ("cell", "cells"),
     ("battery", "batteries"), ("charge", "charging"), ("fire", "fires")],
)
def test_inflections_converge_on_one_stem(a, b):
    assert stem(a) == stem(b)


def test_stemming_is_applied_to_content_terms():
    assert content_terms("venting gases") & content_terms("the cell vents gas")


def test_trailing_citation_markers_stay_with_their_sentence():
    """Regression: a detached "[1]" left cited claims graded as uncited."""
    out = sentences("Overheating begins above 80 °C. [1] Venting follows at 12 bar. [2]")
    assert out == ["Overheating begins above 80 °C. [1]", "Venting follows at 12 bar. [2]"]
