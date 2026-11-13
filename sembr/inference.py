import re
from collections import deque

import numpy as np
from tqdm import trange


DEFAULT_LINE_LENGTH_PENALTY_WEIGHT = 0.05
DEFAULT_MAX_LINE_WIDTH = 80
NEWLINE_TOKEN = '[newline]'
# Sentence-ending punctuation, optionally followed by closing quotes,
# brackets or markup.
SENTENCE_END_RE = re.compile(r'[.!?…]["\'”’)\]*_`]*$')
# The adaptive algorithm prices a break at -log(p), capped at this value,
# and a character beyond the maximum line width at ADAPTIVE_OVERFLOW_COST.
# A line may thus run a few characters over rather than break at a spot
# the model considers unlikely, but never by more than
# ADAPTIVE_MAX_BREAK_COST / ADAPTIVE_OVERFLOW_COST characters if it can break.
ADAPTIVE_MAX_BREAK_COST = 12.0
ADAPTIVE_OVERFLOW_COST = 0.5


def _parse_positive_int(name, value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if ':' in value or '@' in value:
            raise ValueError(
                f'{name} must be an integer.')
        value = int(value)
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive.')
    return value


def _line_length_bounds(
    preferred_min_tokens_per_line=None,
    preferred_max_tokens_per_line=None,
):
    preferred_min_tokens_per_line = _parse_positive_int(
        'preferred_min_tokens_per_line', preferred_min_tokens_per_line)
    preferred_max_tokens_per_line = _parse_positive_int(
        'preferred_max_tokens_per_line', preferred_max_tokens_per_line)

    if (
        preferred_min_tokens_per_line is None
        and preferred_max_tokens_per_line is None
    ):
        return None
    elif preferred_min_tokens_per_line is None:
        preferred_min_tokens_per_line = preferred_max_tokens_per_line
    elif preferred_max_tokens_per_line is None:
        preferred_max_tokens_per_line = preferred_min_tokens_per_line

    if preferred_min_tokens_per_line > preferred_max_tokens_per_line:
        raise ValueError(
            'preferred_min_tokens_per_line must be less than or equal to '
            'preferred_max_tokens_per_line.')
    return preferred_min_tokens_per_line, preferred_max_tokens_per_line


def _line_length_loss(length, lower, upper, weight=1.0):
    if lower <= length <= upper:
        return 0.0
    if length < lower:
        return float(weight * (lower - length) ** 2)
    return float(weight * (length - upper) ** 2)


def _boundary_cost_delta(start, off_costs, break_costs):
    if start == 0:
        return 0.0
    return float(break_costs[start] - off_costs[start])


def _logsumexp(values, axis=None, keepdims=False):
    maximum = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    result = np.log(summed) + maximum
    if keepdims:
        return result
    return np.squeeze(result, axis=axis)


def _log_softmax(values, axis=-1):
    return values - _logsumexp(values, axis=axis, keepdims=True)


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


class _LiChaoLine:
    def __init__(self, slope, intercept, index):
        self.slope = slope
        self.intercept = intercept
        self.index = index

    def value(self, x):
        return self.slope * x + self.intercept


class _LiChaoNode:
    def __init__(self):
        self.line = None
        self.left = None
        self.right = None


class _LiChaoMin:
    def __init__(self, x_left, x_right):
        self.x_left = x_left
        self.x_right = x_right
        self.root = _LiChaoNode()
        self.size = 0

    def add_line(self, line):
        self.size += 1
        self._add_line(self.root, self.x_left, self.x_right, line)

    def _add_line(self, node, left, right, line):
        if node.line is None:
            node.line = line
            return

        mid = (left + right) // 2
        current = node.line
        if line.value(mid) < current.value(mid):
            node.line, line = line, current
            current = node.line
        if left == right:
            return

        if line.value(left) < current.value(left):
            if node.left is None:
                node.left = _LiChaoNode()
            self._add_line(node.left, left, mid, line)
        elif line.value(right) < current.value(right):
            if node.right is None:
                node.right = _LiChaoNode()
            self._add_line(node.right, mid + 1, right, line)

    def query(self, x):
        if self.size == 0:
            return float('inf'), -1
        return self._query(self.root, self.x_left, self.x_right, x)

    def _query(self, node, left, right, x):
        if node is None:
            return float('inf'), -1

        best = (node.line.value(x), node.line.index)
        if left == right:
            return best

        mid = (left + right) // 2
        if x <= mid:
            child_best = self._query(node.left, left, mid, x)
        else:
            child_best = self._query(node.right, mid + 1, right, x)
        if child_best[0] < best[0]:
            return child_best
        return best


def _torch_prepare_batch(data, device):
    return data.to(device)


def _torch_infer_slice(model, data, islice):
    import torch

    inids = data['input_ids'][:, islice]
    attns = data['attention_mask'][:, islice]
    with torch.no_grad():
        outputs = model(
            input_ids=inids, attention_mask=attns, return_dict=True)
    return (
        outputs.logits.detach().float().cpu().numpy(),
        attns.detach().cpu().numpy(),
    )


def _mlx_prepare_batch(data, device):
    return data


def _mlx_infer_slice(model, data, islice):
    import mlx.core as mx

    inids = mx.array(data['input_ids'][:, islice])
    attns = mx.array(data['attention_mask'][:, islice])
    outputs = model(input_ids=inids, attention_mask=attns, return_dict=True)
    mx.eval(outputs.logits)
    logits = np.array(outputs.logits.astype(mx.float32))
    return logits, data['attention_mask'][:, islice]


def _tiled_inference_with_backend(
    model, collator, results, batch_size, overlap_divisor,
    *, device, prepare_batch, infer_slice,
):
    max_length = model.config.max_position_embeddings
    overlap_length = int(max_length / overlap_divisor)
    input_ids = [{'input_ids': r['input_ids']} for r in results]
    num_paras = len(input_ids)
    lengths = [len(i['input_ids']) for i in input_ids]
    sorted_indices = sorted(
        range(num_paras), key=lambda i: lengths[i], reverse=True)
    logits = np.zeros(
        (num_paras, max(lengths), model.config.num_labels), dtype=np.float32)
    counts = np.zeros((num_paras, max(lengths)), dtype=np.int64)
    for b in trange(0, num_paras, batch_size):
        bslice = slice(b, min(num_paras, b + batch_size))
        bindices = sorted_indices[bslice]
        binids = [input_ids[i] for i in bindices]
        data = prepare_batch(collator(binids, return_tensors='pt'), device)
        num_tokens = data['input_ids'].shape[1]
        for i in range(0, num_tokens, max_length - overlap_length):
            islice = slice(i, min(num_tokens, i + max_length))
            slice_logits, slice_counts = infer_slice(model, data, islice)
            logits[bindices, islice] += slice_logits
            counts[bindices, islice] += slice_counts
    attns = counts > 0
    np.divide(
        logits,
        counts[..., None],
        out=logits,
        where=counts[..., None] > 0,
    )
    logits[~attns] = 0
    return logits, attns


def _tiled_inference(model, collator, results, batch_size, overlap_divisor):
    if getattr(model, 'device', None) == 'mlx':
        device = 'cpu'
        prepare_batch = _mlx_prepare_batch
        infer_slice = _mlx_infer_slice
    else:
        device = model.device
        prepare_batch = _torch_prepare_batch
        infer_slice = _torch_infer_slice
    return _tiled_inference_with_backend(
        model, collator, results, batch_size, overlap_divisor,
        device=device,
        prepare_batch=prepare_batch,
        infer_slice=infer_slice,
    )


def _format_labels(id2label, preds, attns, results):
    modes, indents = [], []
    for i, (p, a) in enumerate(zip(preds, attns)):
        para_modes, para_indents = [], []
        for name in [id2label[int(t)] for t in p[a]]:
            if name == 'off':
                mode, indent = 'off', 0
            else:
                mode, indent = name.split('-')
            para_modes.append(mode)
            para_indents.append(int(indent))
        modes.append(para_modes)
        indents.append(para_indents)
    for r, m, i in zip(results, modes, indents):
        r['modes'] = m
        r['indents'] = i
    return results


def predict_argmax(logits, counts, **kwargs):
    logits = _as_numpy(logits)
    return logits.argmax(axis=2)


def predict_logit_adjustment(logits, counts, **kwargs):
    logits = _as_numpy(logits).copy()
    delta = 1.0
    logits[:, :, 0] -= delta
    logits[:, :, 1:] += delta / logits.shape[2]
    return logits.argmax(axis=2)


def predict_greedy_linebreaks(
    logits, counts, preferred_max_tokens_per_line=None, **kwargs
):
    logits = _as_numpy(logits).copy()
    counts = _as_numpy(counts)
    bounds = _line_length_bounds(
        preferred_max_tokens_per_line=preferred_max_tokens_per_line,
    )
    if bounds is None:
        return logits.argmax(axis=2)
    _, preferred_max_tokens_per_line = bounds
    has_long_lines = True
    while has_long_lines:
        has_long_lines = False
        for b in range(logits.shape[0]):
            row_preds = logits[b].argmax(axis=1)
            if row_preds.size == 0:
                continue
            starts = np.r_[0, np.nonzero(row_preds[1:] != row_preds[:-1])[0] + 1]
            stops = np.r_[starts[1:], row_preds.size]
            modes = row_preds[starts]
            repeats = stops - starts
            long_lines = (
                (modes == 0) & (repeats > preferred_max_tokens_per_line)
            )
            if long_lines.any():
                has_long_lines = True
            starts, stops = starts[long_lines], stops[long_lines]
            for s, e in zip(starts, stops):
                # find the token with the lowest "off" logit
                max_index = logits[b, s:e, 0].argmin(0)
                # force linebreak by reducing "off" by -1e6
                logits[b, s + max_index, 0] -= 1e6
    return logits.argmax(axis=2)


def predict_balanced_linebreaks(
    logits, counts,
    preferred_min_tokens_per_line=None,
    preferred_max_tokens_per_line=None,
    line_length_penalty_weight=DEFAULT_LINE_LENGTH_PENALTY_WEIGHT,
    **kwargs
):
    logits = _as_numpy(logits)
    counts = _as_numpy(counts)
    bounds = _line_length_bounds(
        preferred_min_tokens_per_line=preferred_min_tokens_per_line,
        preferred_max_tokens_per_line=preferred_max_tokens_per_line,
    )
    if bounds is None:
        return logits.argmax(axis=2)
    lower, upper = bounds
    if line_length_penalty_weight < 0:
        raise ValueError('line_length_penalty_weight must be non-negative.')

    preds = np.zeros(logits.shape[:2], dtype=np.int64)
    for b in range(logits.shape[0]):
        num_tokens = int(counts[b].sum().item())
        if num_tokens == 0:
            continue

        row_logits = logits[b, :num_tokens]
        break_labels = row_logits[:, 1:].argmax(axis=1) + 1
        log_probs = _log_softmax(row_logits, axis=1)
        off_costs = -log_probs[:, 0]
        break_costs = -_logsumexp(log_probs[:, 1:], axis=1)
        off_cost_prefix = np.r_[0.0, np.cumsum(off_costs)]

        dp = [float('inf')] * (num_tokens + 1)
        prev = [-1] * (num_tokens + 1)
        values = [float('inf')] * (num_tokens + 1)
        range_min = deque()
        long_hull = _LiChaoMin(0, max(0, num_tokens - upper))
        dp[0] = 0.0
        values[0] = 0.0
        for end in range(1, num_tokens + 1):
            range_start = end - lower
            if range_start >= 0:
                while (
                    range_min
                    and values[range_min[-1]] > values[range_start]
                ):
                    range_min.pop()
                range_min.append(range_start)

            long_start = end - upper - 1
            if long_start >= 0:
                line = _LiChaoLine(
                    -2.0 * line_length_penalty_weight * long_start,
                    (
                        values[long_start]
                        + line_length_penalty_weight * long_start ** 2
                    ),
                    long_start,
                )
                long_hull.add_line(line)

            range_stop = end - upper
            while range_min and range_min[0] < range_stop:
                range_min.popleft()

            candidates = []
            for start in range(max(0, end - lower + 1), end):
                length = end - start
                loss = values[start] + _line_length_loss(
                    length, lower, upper, line_length_penalty_weight)
                candidates.append((loss, start))

            if range_min:
                start = range_min[0]
                candidates.append((values[start], start))

            if long_hull.size:
                x = end - upper
                hull_value, start = long_hull.query(x)
                candidates.append((
                    hull_value + line_length_penalty_weight * x ** 2,
                    start,
                ))

            best_value, best_start = min(candidates)
            dp[end] = best_value + float(off_cost_prefix[end].item())
            prev[end] = best_start
            values[end] = dp[end] - float(off_cost_prefix[end].item())
            if end < num_tokens:
                values[end] += _boundary_cost_delta(
                    end, off_costs, break_costs)

        start = prev[num_tokens]
        while start > 0:
            preds[b, start] = break_labels[start]
            start = prev[start]

    return preds


def _adaptive_breaks(
    words, break_costs, allowed, max_line_width,
    first_line_width=0, continuation_width=0, line_starts=None,
):
    """
    Choose the words to break before:
    always where the model agrees with a sentence end
    or with a line break of the input (the words in line_starts),
    and otherwise only where a line would exceed max_line_width,
    at the breaks the model finds most likely.
    """
    num_words = len(words)
    prefix = [0]
    for word in words:
        prefix.append(
            prefix[-1] + (0 if word == NEWLINE_TOKEN else len(word)))

    def width(start, stop):
        offset = first_line_width if start == 0 else continuation_width
        leading = len(words[start]) - len(words[start].lstrip())
        return offset + prefix[stop] - prefix[start] - leading

    candidates = [i for i in range(1, num_words) if allowed[i]]
    breaks = {
        i for i in candidates
        if break_costs[i] <= -np.log(0.5) and (
            (line_starts is not None and line_starts[i])
            or SENTENCE_END_RE.search(
                ''.join(words[max(0, i - 8):i]).rstrip()))}
    fixed = {0, num_words} | breaks
    fixed.update(
        i + 1 for i, word in enumerate(words[:-1]) if word == NEWLINE_TOKEN)
    fixed = sorted(fixed)

    max_overflow = ADAPTIVE_MAX_BREAK_COST / ADAPTIVE_OVERFLOW_COST
    for start, stop in zip(fixed, fixed[1:]):
        nodes = [start, *(i for i in candidates if start < i < stop), stop]
        best = [0.0] + [float('inf')] * (len(nodes) - 1)
        prev = [0] * len(nodes)
        for j in range(1, len(nodes)):
            for i in range(j - 1, -1, -1):
                overflow = max(0, width(nodes[i], nodes[j]) - max_line_width)
                # Longer lines only overflow more; the shortest is always
                # considered, as it may be a single unbreakable word.
                if overflow > max_overflow and i < j - 1:
                    break
                cost = best[i] + overflow * ADAPTIVE_OVERFLOW_COST
                if cost < best[j]:
                    best[j], prev[j] = cost, i
            best[j] += break_costs[nodes[j]] if j < len(nodes) - 1 else 0.0
        j = prev[-1]
        while j > 0:
            breaks.add(nodes[j])
            j = prev[j]
    return sorted(breaks)


def predict_adaptive_linebreaks(
    logits, counts, results=None, id2label=None,
    max_line_width=DEFAULT_MAX_LINE_WIDTH, **kwargs
):
    if results is None or id2label is None:
        raise ValueError(
            'The adaptive algorithm requires results and id2label.')
    logits = _as_numpy(logits)
    counts = _as_numpy(counts)
    # Breaks are only placed at existing whitespace, never with a "nospace"
    # label, which would add a comment character to hide the line break.
    space_labels = np.array(sorted(
        i for i, name in id2label.items() if name.startswith('space-')))
    preds = np.zeros(logits.shape[:2], dtype=np.int64)
    for b, result in enumerate(results):
        words = result['words']
        if not words:
            continue
        row_logits = logits[b, :len(words)]
        break_costs = np.minimum(
            -_logsumexp(_log_softmax(row_logits, axis=1)[:, 1:], axis=1),
            ADAPTIVE_MAX_BREAK_COST)
        allowed = result.get('break_allowed')
        if allowed is None:
            allowed = [
                i > 0 and word != word.lstrip()
                for i, word in enumerate(words)]
        breaks = _adaptive_breaks(
            words, break_costs.tolist(), allowed, max_line_width,
            result.get('first_line_width', 0),
            result.get('continuation_width', 0),
            [mode != 'off' for mode in result['modes']])
        for i in breaks:
            preds[b, i] = space_labels[row_logits[i, space_labels].argmax()]
    return preds


PREDICT_FUNC_MAP = {
    'argmax': predict_argmax,
    'logit_adjustment': predict_logit_adjustment,
    'greedy_linebreaks': predict_greedy_linebreaks,
    'balanced_linebreaks': predict_balanced_linebreaks,
    'adaptive': predict_adaptive_linebreaks,
}

# Algorithms that decide every line break themselves,
# so processors should not carry over the existing ones.
REFLOW_PREDICT_FUNCS = {'adaptive'}


def inference(
    text, tokenizer, model, processor,
    predict_func='argmax', batch_size=8, overlap_divisor=8,
    *,
    preferred_min_tokens_per_line=None, preferred_max_tokens_per_line=None,
    line_length_penalty_weight=DEFAULT_LINE_LENGTH_PENALTY_WEIGHT,
    max_line_width=DEFAULT_MAX_LINE_WIDTH,
):
    if text.strip() == '':
        return []
    processor.reflow = predict_func in REFLOW_PREDICT_FUNCS

    if getattr(model, 'device', None) == 'mlx':
        from .mlx import NumpyTokenClassificationCollator
        collator = NumpyTokenClassificationCollator(
            tokenizer, padding='longest')
    else:
        from transformers import DataCollatorForTokenClassification
        collator = DataCollatorForTokenClassification(
            tokenizer, padding='longest')
    results = processor.parse_text(text, split=isinstance(text, str))
    results = processor.tokenize_with_modes(tokenizer, results)
    logits, counts = _tiled_inference(
        model, collator, results, batch_size, overlap_divisor)
    preds = PREDICT_FUNC_MAP[predict_func](
        logits,
        counts,
        preferred_min_tokens_per_line=preferred_min_tokens_per_line,
        preferred_max_tokens_per_line=preferred_max_tokens_per_line,
        line_length_penalty_weight=line_length_penalty_weight,
        max_line_width=max_line_width,
        results=results,
        id2label=model.config.id2label,
    )
    return _format_labels(model.config.id2label, preds, counts, results)


def sembr(
    text, tokenizer, model, processor,
    predict_func='argmax', batch_size=8, overlap_divisor=8,
    *,
    preferred_min_tokens_per_line=None, preferred_max_tokens_per_line=None,
    line_length_penalty_weight=DEFAULT_LINE_LENGTH_PENALTY_WEIGHT,
    max_line_width=DEFAULT_MAX_LINE_WIDTH,
    **_,
):
    results = inference(
        text, tokenizer, model, processor, predict_func,
        batch_size, overlap_divisor,
        preferred_min_tokens_per_line=preferred_min_tokens_per_line,
        preferred_max_tokens_per_line=preferred_max_tokens_per_line,
        line_length_penalty_weight=line_length_penalty_weight,
        max_line_width=max_line_width)
    return processor.generate(results, join=True)
