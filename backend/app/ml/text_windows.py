"""Tokenizer-aligned windows preserving every source character."""


def text_windows(tokenizer, text: str, budget: int) -> list[str]:
    if not text:
        return [text]
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= budget:
        return [text]
    windows = []
    start = 0
    for index in range(budget, len(offsets), budget):
        end = offsets[index][0]
        if end > start:
            windows.append(text[start:end])
            start = end
    windows.append(text[start:])
    # Retokenization at a new boundary can change token counts. Split further
    # when needed rather than letting the model silently truncate the window.
    bounded = []
    pending = list(reversed(windows))
    while pending:
        window = pending.pop()
        count = len(tokenizer(window, add_special_tokens=False, truncation=False,
                              return_offsets_mapping=True)["offset_mapping"])
        if count <= budget:
            bounded.append(window)
        elif len(window) > 1:
            midpoint = len(window) // 2
            pending.extend([window[midpoint:], window[:midpoint]])
        else:
            raise ValueError("A source character exceeds the model token budget")
    return bounded
