from docstruct.agent.builder import DocumentBuilder
from docstruct.agent.ops import NewHeading, NewParagraph, NewTable


def test_paragraph_before_any_heading_goes_to_preamble():
    b = DocumentBuilder()
    b.apply(NewParagraph(text="مقدمه"))

    doc = b.build()

    assert doc.preamble_paragraphs == ["مقدمه"]
    assert doc.sections == []


def test_heading_then_paragraph_nests_under_it():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewParagraph(text="متن"))

    doc = b.build()

    assert len(doc.sections) == 1
    assert doc.sections[0].heading == "فصل اول"
    assert doc.sections[0].paragraphs == ["متن"]


def test_deeper_heading_nests_as_subsection():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewHeading(level=2, text="بخش ۱-۱"))
    b.apply(NewParagraph(text="متن تودرتو"))

    doc = b.build()

    assert len(doc.sections) == 1
    assert len(doc.sections[0].subsections) == 1
    sub = doc.sections[0].subsections[0]
    assert sub.heading == "بخش ۱-۱"
    assert sub.paragraphs == ["متن تودرتو"]


def test_sibling_heading_at_same_level_pops_back_up():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewHeading(level=2, text="بخش ۱-۱"))
    b.apply(NewHeading(level=1, text="فصل دوم"))
    b.apply(NewParagraph(text="متن فصل دوم"))

    doc = b.build()

    assert len(doc.sections) == 2
    assert doc.sections[0].subsections[0].heading == "بخش ۱-۱"
    assert doc.sections[1].heading == "فصل دوم"
    assert doc.sections[1].paragraphs == ["متن فصل دوم"]


def test_heading_level_can_jump_without_intermediate_levels():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewHeading(level=3, text="زیربخش عمیق"))

    doc = b.build()

    assert doc.sections[0].subsections[0].heading == "زیربخش عمیق"
    assert doc.sections[0].subsections[0].level == 3


def test_table_attaches_to_open_section_or_preamble():
    b = DocumentBuilder()
    b.apply(NewTable(headers=["a"], rows=[["1"]]))
    b.apply(NewHeading(level=1, text="فصل"))
    b.apply(NewTable(caption="جدول", headers=["b"], rows=[["2"]]))

    doc = b.build()

    assert doc.preamble_tables[0].rows == [["1"]]
    assert doc.sections[0].tables[0].caption == "جدول"


def test_set_title_only_takes_first_non_empty_value():
    b = DocumentBuilder()
    b.set_title("عنوان اول")
    b.set_title("عنوان دوم")

    assert b.build().title == "عنوان اول"


def test_duplicate_op_within_dedup_window_is_dropped():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewParagraph(text="یک پاراگراف"))
    # a chunk-overlap re-emission of the same paragraph, right behind it
    b.apply(NewParagraph(text="یک پاراگراف"))
    b.apply(NewParagraph(text="پاراگراف بعدی"))

    doc = b.build()

    assert doc.sections[0].paragraphs == ["یک پاراگراف", "پاراگراف بعدی"]


def test_duplicate_heading_is_dropped_and_open_section_stays_the_same():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewHeading(level=1, text="فصل اول"))  # re-emitted, e.g. from overlap
    b.apply(NewParagraph(text="متن"))

    doc = b.build()

    # not duplicated into two sibling sections — the paragraph lands under the one section
    assert len(doc.sections) == 1
    assert doc.sections[0].paragraphs == ["متن"]


def test_maddeh_heading_is_forced_to_match_its_still_open_sibling_level():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="دستورالعمل"))
    b.apply(NewHeading(level=4, text="ماده1 - تعاریف :"))
    b.apply(NewHeading(level=6, text="بند ۱-۲"))  # nested deeper under ماده1
    # model mis-levels this as 5 instead of 4, but it's still a ماده sibling reachable in the open chain
    b.apply(NewHeading(level=5, text="ماده2 - ترکیب :"))
    b.apply(NewParagraph(text="متن ماده۲"))

    doc = b.build()

    maddeh1 = doc.sections[0].subsections[0]
    maddeh2 = doc.sections[0].subsections[1]
    assert maddeh1.heading == "ماده1 - تعاریف :"
    assert maddeh2.heading == "ماده2 - ترکیب :"
    assert maddeh1.level == maddeh2.level == 4
    assert maddeh2.paragraphs == ["متن ماده۲"]


def test_maddeh_heading_keeps_its_own_level_when_no_sibling_is_open():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="دستورالعمل اول"))
    b.apply(NewHeading(level=4, text="ماده1 - قدیمی :"))
    b.apply(NewHeading(level=1, text="دستورالعمل دوم"))  # closes out دستورالعمل اول entirely
    b.apply(NewHeading(level=3, text="ماده1 - جدید :"))

    doc = b.build()

    assert doc.sections[1].subsections[0].heading == "ماده1 - جدید :"
    assert doc.sections[1].subsections[0].level == 3


def test_breadcrumb_reflects_currently_open_path():
    b = DocumentBuilder()
    b.apply(NewHeading(level=1, text="فصل اول"))
    b.apply(NewHeading(level=2, text="بخش ۱-۱"))

    assert b.breadcrumb() == [(1, "فصل اول"), (2, "بخش ۱-۱")]

    b.apply(NewHeading(level=1, text="فصل دوم"))

    assert b.breadcrumb() == [(1, "فصل دوم")]
