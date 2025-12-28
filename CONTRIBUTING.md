# Contributing to AI Chatbot

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Bugs

- Use GitHub Issues
- Include: Python version, error message, steps to reproduce
- Provide screenshots if UI-related

### Suggesting Features

- Open an issue with [FEATURE REQUEST] tag
- Describe the feature and its use case
- Explain why it would be valuable

### Pull Requests

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Make your changes
4. Write/update tests
5. Update documentation
6. Commit with clear messages
   ```bash
   git commit -m "feat: add amazing feature"
   ```
7. Push and create PR

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions focused
- Add comments for complex logic

### Commit Messages

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance

### Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Lint code
flake8 src/
black src/
```

### Documentation

- Update README.md for new features
- Add docstrings to new functions
- Update API.md if adding endpoints
- Include usage examples

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-chatbot.git

# Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run in development mode
streamlit run src/app.py
```

## Code Review Process

1. Maintainers review PRs within 48 hours
2. Address feedback promptly
3. Keep PRs focused and small
4. Once approved, we'll merge

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Give constructive feedback
- Credit others' work

## Questions?

- Open a Discussion on GitHub
- Email: support@example.com
- Join our Discord (coming soon)

## License

By contributing, you agree your contributions will be licensed under MIT License.

---

**Thank you for making this project better! 🚀**
