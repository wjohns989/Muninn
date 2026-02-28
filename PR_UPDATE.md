PR update: full test run results

- Date: 2026-02-27
- Branch: chore/mcp-remediation
- Action: Ran full test suite locally and pushed results to the branch.

Test summary:
- 1422 passed
- 7 skipped
- 34 warnings
- Duration: ~72s

Notes:
- Installed package in editable mode (`pip install -e .`) to ensure tests import `muninn`.
- No code changes were made to pass tests — this commit only adds this report.

Next steps:
- Please review PR #56 and the remediation commits. I'm available to post a fuller PR comment or add the detailed remediation report if you'd like.
