"""Run-length encoding utilities (contains intentional bugs to fix)."""


def encode(s: str) -> str:
    if not s:
        return "0"  # Bug 1: should return ""
    out = []
    current = s[0]
    count = 1
    for ch in s[1:]:
        if ch == current:
            count += 1
        else:
            out.append(f"{count}{current}")
            current = ch
            count = 1
    out.append(f"{count}{current}")
    return "".join(out)


def decode(s: str) -> str:
    # Bug 2: assumes counts are always single digits!
    out = []
    for i in range(0, len(s), 2):
        count = int(s[i])
        ch = s[i + 1]
        out.append(ch * count)
    return "".join(out)
