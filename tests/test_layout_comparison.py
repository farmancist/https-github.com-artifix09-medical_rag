from scripts.compare_pdf_layout import sample_page_indices


def test_sample_page_indices_are_deterministic_and_distributed() -> None:
    assert sample_page_indices(100, 5) == [0, 25, 50, 74, 99]


def test_sample_page_indices_return_all_when_sample_covers_document() -> None:
    assert sample_page_indices(3, 10) == [0, 1, 2]


def test_single_page_sample_uses_first_page() -> None:
    assert sample_page_indices(10, 1) == [0]
