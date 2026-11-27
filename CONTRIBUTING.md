# Contributing to cv-benchkit

Thank you for your interest in contributing to **cv-benchkit**! We welcome community contributions of all kinds: bug fixes, new bounding-box converters, metrics, documentation improvements, and additional unit tests.

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please be kind, respectful, and constructive in all discussions, pull requests, and issue threads.

---

## Finding & Claiming Issues

We keep a curated list of newcomer-friendly tasks labeled [`good first issue`](https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

- **How to Claim an Issue:** Post a short comment on an unclaimed `good first issue` stating your plan, and wait for issue assignment before opening a PR so contributors do not duplicate work. Include `Closes #<issue-number>` (or `Fixes #<issue-number>`) in your PR description so the issue auto-closes when merged.
- **First-Time Contributor Priority:** Issues labeled `good first issue` prioritize contributors with 0 prior merged PRs on `cv-benchkit`. Repeat contributors are welcome up to 2 merged PRs per contributor so every newcomer gets a fair shot.

---

## Development Setup

We use [`uv`](https://github.com/astral-sh/uv) for fast and reproducible environment and dependency management.

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/cv-benchkit.git
   cd cv-benchkit
   git remote add upstream https://github.com/udayraj1238/cv-benchkit.git
   ```

2. **Install dependencies and pre-commit hooks**:
   ```bash
   uv sync --extra dev
   uv run pre-commit install
   ```

3. **Create a topic branch**:
   ```bash
   git checkout -b feat/add-yolo-obb-support
   # or
   git checkout -b fix/empty-batch-giou
   ```

---

## Coding Standards

All code submitted to `cv-benchkit` must meet these standards:

- **Type Annotations**: All public and internal functions must be 100% typed and pass `uv run mypy src/` in strict mode.
- **Linting & Formatting**: We enforce [`ruff`](https://github.com/astral-sh/ruff) for linting and formatting. Run:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  ```
- **Testing**: Every new feature, converter, or bug fix must come with unit tests in `tests/`. We maintain >=95% test coverage.
  ```bash
  uv run pytest
  ```
- **Commit Messages**: We follow Conventional Commits:
  - `feat: add rotated bounding box (OBB) converter`
  - `fix: handle zero-width box in GIoU computation`
  - `docs: update quick start instructions in README`
  - `test: add edge cases for empty annotation sets`

---

## Submitting a Pull Request

1. Ensure all local quality gates pass:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src/
   uv run pytest
   ```
2. Push your branch to your fork:
   ```bash
   git push origin <your-branch-name>
   ```
3. Open a Pull Request against `main`. Follow our PR template and reference any relevant issues (e.g., `Fixes #12`).

---

## Need an Idea?

Check out our [Good First Issues Catalog](docs/GOOD_FIRST_ISSUES_CATALOG.md) which contains 40 curated, bite-sized tasks ready to claim and build!
