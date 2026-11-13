from sembr.processors.markdown import MarkdownProcessor


def _reflow_processor():
    processor = MarkdownProcessor(spaces=2)
    processor.reflow = True
    return processor


def _flat_texts(processor, text):
    return [r['flat_lines'] for r in processor.parse_text(text)]


def test_reflow_joins_wrapped_paragraph_lines():
    text = (
        'Top level\nparagraph.\n\n'
        '> Quoted\n> paragraph\nlazily continued.\n\n'
        '- A list\n  item\n  - nested\n    item\n'
    )

    assert _flat_texts(_reflow_processor(), text) == [
        'Top level paragraph.[newline]',
        'Quoted paragraph lazily continued.[newline]',
        'A list item[newline]',
        'nested item[newline]',
    ]


def test_reflow_leaves_headings_and_hard_line_breaks_out():
    text = (
        '# Heading\n\n'
        'Setext\n======\n\n'
        'Two spaces  \nhard break.\n\n'
        'Backslash\\\nhard break.\n\n'
        'Plain\nparagraph.\n'
    )

    assert _flat_texts(_reflow_processor(), text) == [
        'Plain paragraph.[newline]']


def test_reflow_measures_first_and_continuation_line_offsets():
    region, = _reflow_processor().parse_text('> 1. Ünïcödé item\n>    text.\n')

    assert region['original_text'] == 'Ünïcödé item\n>    text.'
    assert region['first_line_width'] == 5
    assert region['continuation_width'] == len('   > ')


def test_break_allowed_only_at_safe_whitespace():
    words = [
        'See', ' `a', ' b`', ' and', ' [a', ' link', '](', 'u', ')', ' or',
        ' -', ' 1', '.', ' x', '-', 'y', ' *', 'em', '*', ' >', ' q',
    ]

    allowed = _reflow_processor()._break_allowed(words)

    assert [w for w, a in zip(words, allowed) if a] == [
        ' `a', ' and', ' [a', ' or', ' x', ' *', ' q']


def test_reflow_generates_unindented_continuation_lines():
    processor = _reflow_processor()
    text = '- First sentence. Second\n  sentence.\n'
    region, = processor.parse_text(text)
    region['words'] = ['First', ' sentence', '.', ' Second', ' sentence', '.', '[newline]']
    region['modes'] = ['off', 'off', 'off', 'space', 'off', 'off', 'off']
    # Indent levels predicted by the model are not applied.
    region['indents'] = [0, 0, 0, 1, 0, 0, 0]

    assert processor.generate([region]) == (
        '- First sentence.\n  Second sentence.\n')
