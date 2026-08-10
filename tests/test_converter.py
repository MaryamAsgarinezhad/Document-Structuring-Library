from docstruct.converter import structured_document_to_markdown
from docstruct.schema import Section, StructuredDocument, TableData


def test_title_and_heading_levels():
    doc = StructuredDocument(
        title="عنوان سند",
        sections=[
            Section(
                heading="فصل اول",
                level=1,
                paragraphs=["پاراگراف."],
                subsections=[Section(heading="بخش ۱-۱", level=2, paragraphs=["زیرمتن."])],
            )
        ],
    )

    md = structured_document_to_markdown(doc)

    assert "# عنوان سند" in md
    assert "## فصل اول" in md
    assert "### بخش ۱-۱" in md
    title_pos = md.index("# عنوان سند")
    ch1_pos = md.index("## فصل اول")
    sub_pos = md.index("### بخش ۱-۱")
    assert title_pos < ch1_pos < sub_pos


def test_top_level_section_is_h2_even_without_title():
    doc = StructuredDocument(sections=[Section(heading="فصل اول", level=1)])

    md = structured_document_to_markdown(doc)

    assert md.startswith("## فصل اول")


def test_reading_order_preserved_across_siblings():
    doc = StructuredDocument(
        sections=[
            Section(heading="اول", level=1, paragraphs=["A"]),
            Section(heading="دوم", level=1, paragraphs=["B"]),
        ]
    )

    md = structured_document_to_markdown(doc)

    assert md.index("اول") < md.index("دوم")


def test_table_renders_as_gfm_pipe_table():
    table = TableData(caption="جدول نرخ‌ها", headers=["ردیف", "نرخ"], rows=[["۱", "۱۰٪"], ["۲", "۲۰٪"]])
    doc = StructuredDocument(sections=[Section(heading="آمار", level=1, tables=[table])])

    md = structured_document_to_markdown(doc)

    assert "| ردیف | نرخ |" in md
    assert "| --- | --- |" in md
    assert "| ۱ | ۱۰٪ |" in md
    assert "*جدول نرخ‌ها*" in md


def test_table_cell_pipe_and_newline_are_escaped():
    table = TableData(headers=["a"], rows=[["x|y\nz"]])
    doc = StructuredDocument(sections=[Section(heading="h", level=1, tables=[table])])

    md = structured_document_to_markdown(doc)

    assert "x\\|y z" in md
    assert "x|y\nz" not in md


def test_preamble_appears_before_sections():
    doc = StructuredDocument(
        title="سند",
        preamble_paragraphs=["مقدمه"],
        sections=[Section(heading="فصل", level=1, paragraphs=["بدنه"])],
    )

    md = structured_document_to_markdown(doc)

    assert md.index("مقدمه") < md.index("فصل")


def test_ragged_table_row_padded_to_header_width():
    table = TableData(headers=["a", "b"], rows=[["only-one"]])
    doc = StructuredDocument(sections=[Section(heading="h", level=1, tables=[table])])

    md = structured_document_to_markdown(doc)

    assert "| only-one |  |" in md


def test_empty_document_renders_to_empty_string():
    md = structured_document_to_markdown(StructuredDocument())

    assert md == ""
