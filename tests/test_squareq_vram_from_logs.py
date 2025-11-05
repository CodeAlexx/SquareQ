import re

BF16_SAMPLE = "[VRAM:after_forward] alloc=22.34GB peak=23.10GB"
BP8_SAMPLE = "[VRAM:after_forward] alloc=11.20GB peak=11.40GB"


def parse_peak(line: str) -> float:
    match = re.search(r"peak=(\d+\.\d+)GB", line)
    assert match, f"no peak in line: {line}"
    return float(match.group(1))


def test_bp8_peak_lower_than_bf16():
    assert parse_peak(BP8_SAMPLE) < parse_peak(BF16_SAMPLE)
