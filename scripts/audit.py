"""Lightweight automated auditor for thesis code consistency.

Run manually: python scripts/audit.py
Or automatically via pre-commit hooks.
"""

import re
import sys
from pathlib import Path

CANONICAL_NAMES = {
    r'\bA_s\b': 'A_sym',
    r'\battn_sym\b': 'A_sym',
    r'\bsymmetric_attention\b': 'A_sym',
    r'\bA_a\b': 'A_anti',
    r'\battn_anti\b': 'A_anti',
    r'\bantisymmetric_attention\b': 'A_anti',
    r'\bsize_diff\b': 'delta_size',
    r'\babs_size_gap\b': 'delta_size',
    r'\bstyle_diff\b': 'delta_style',
    r'\bnext_ret\b': 'ret_next',
}

MATH_OPS = [
    r'torch\.matmul',
    r'torch\.mm',
    r'\s@\s',
    r'\.transpose',
    r'0\.5\s*\*\s*\(',
]

EQUATION_PATTERN = r'#.*[Ee]q[\.\s]*\(?\d+\)?|#.*Kelly.*\d{4}'


def check_naming(filepath: Path) -> list[str]:
    """Check for non-canonical variable names."""
    issues = []
    content = filepath.read_text()
    for pattern, canonical in CANONICAL_NAMES.items():
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(
                f"WARNING {filepath}:{line_num} - Use '{canonical}' instead of "
                f"'{match.group()}'"
            )
    return issues


def check_equation_refs(filepath: Path) -> list[str]:
    """Check that math operations have equation references nearby."""
    issues = []
    lines = filepath.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        for op_pattern in MATH_OPS:
            if re.search(op_pattern, line):
                context = '\n'.join(lines[max(0, i - 4):i + 1])
                if not re.search(EQUATION_PATTERN, context):
                    issues.append(
                        f"CRITICAL {filepath}:{i} - Math operation without equation "
                        f"reference: {line.strip()[:60]}"
                    )
                break
    return issues


def check_hardcoded_seeds(filepath: Path) -> list[str]:
    """Check for hardcoded random seeds (should come from config)."""
    issues = []
    content = filepath.read_text()
    seed_patterns = [
        (r'random\.seed\(\d+\)', 'random.seed'),
        (r'manual_seed\(\d+\)', 'torch.manual_seed'),
        (r'np\.random\.seed\(\d+\)', 'np.random.seed'),
    ]
    for pattern, name in seed_patterns:
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(
                f"WARNING {filepath}:{line_num} - Hardcoded {name}. "
                f"Use config seed instead."
            )
    return issues


def main() -> int:
    """Run all audits on thesis source directories."""
    all_issues: list[str] = []
    scan_dirs = ['models', 'analysis', 'training', 'data_pipeline', 'visualization']

    for scan_dir in scan_dirs:
        dir_path = Path(scan_dir)
        if not dir_path.exists():
            continue
        for filepath in dir_path.rglob('*.py'):
            all_issues.extend(check_naming(filepath))
            all_issues.extend(check_equation_refs(filepath))
            all_issues.extend(check_hardcoded_seeds(filepath))

    if all_issues:
        print(f"\n{'=' * 60}")
        print(f"THESIS AUDITOR - {len(all_issues)} issues found")
        print(f"{'=' * 60}\n")
        for issue in sorted(all_issues):
            print(f"  {issue}")
        print(f"\n{'=' * 60}\n")
        return 1
    else:
        print("Thesis auditor: all checks passed.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
