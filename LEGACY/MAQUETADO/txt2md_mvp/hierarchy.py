
from typing import List, Dict, Any, Tuple

LEVEL_ORDER = {"h1": 1, "h2": 2, "h3": 3}

def enforce(stack: List[str], new_level: str) -> Tuple[List[str], str]:
    # Ensure we don't jump down by more than one level; repair if needed.
    if not stack:
        # The first heading of a fragment determines the starting level.
        # Don't force it to be h1, as the fragment might be a chapter.
        return [new_level], new_level
    want = LEVEL_ORDER[new_level]
    top = LEVEL_ORDER[stack[-1]]
    if want == top:
        return stack, new_level
    if want == top + 1:
        stack.append(new_level)
        return stack, new_level
    if want <= top:
        # climb up
        while stack and LEVEL_ORDER[stack[-1]] > want:
            stack.pop()
        if not stack or LEVEL_ORDER[stack[-1]] < want:
            stack.append(new_level)
        return stack, new_level
    # Jump too far (e.g., h1 -> h3): repair to intermediate (h2)
    repaired = "h2" if new_level == "h3" else new_level
    if LEVEL_ORDER[repaired] == top + 1:
        stack.append(repaired)
        return stack, repaired
    return stack, new_level
