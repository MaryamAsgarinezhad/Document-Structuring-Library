from docstruct.schema import Section, StructuredDocument, TableData


def test_nested_sections_round_trip_through_json():
    doc = StructuredDocument(
        title="سند نمونه",
        sections=[
            Section(
                heading="فصل اول",
                level=1,
                paragraphs=["متن پاراگراف اول"],
                subsections=[
                    Section(heading="بخش ۱-۱", level=2, paragraphs=["متن تودرتو"]),
                ],
            )
        ],
    )

    restored = StructuredDocument.model_validate_json(doc.model_dump_json())

    assert restored == doc
    assert restored.sections[0].subsections[0].heading == "بخش ۱-۱"


def test_defaults_are_empty_not_shared_mutable_state():
    a = StructuredDocument()
    b = StructuredDocument()

    a.sections.append(Section(heading="x", level=1))

    assert b.sections == []


def test_table_data_defaults():
    table = TableData()

    assert table.caption is None
    assert table.headers == []
    assert table.rows == []
