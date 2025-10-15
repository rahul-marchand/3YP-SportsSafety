# 3YP-SportsSafety

## Project Structure

```
src/
├── eye_tracking/         # ML for eye tracking
├── statistical_analysis/ # Statistical analysis module
├── electronics/          # Electronics code
└── mobile_app/          # App module
tests/                   # Test directory
```

## Getting Started

**First-time setup:**
```bash
# Install dependencies and pre-commit hooks
uv sync --dev
uv run pre-commit install
```

## Contributing

**Main branch must always be working** - never push broken code to main, start adding new features by working on new branches then pull request into main

Example workflow:
1. `git checkout -b my-new-feature`
2. Make your changes and test them
3. `git push` and create a pull request
4. We all review and merge

**Dependency management:**
- Add packages: `uv add <package-name>`
- Add dev tools: `uv add --dev <package-name>`
- Run code: `uv run <script.py>`

**Code quality (automatic with pre-commit):**
- Format code: `uv run ruff format .`
- Check for issues: `uv run ruff check .`
- Fix auto-fixable issues: `uv run ruff check --fix .`

Pre-commit hooks run automatically before each commit to ensure code quality.
