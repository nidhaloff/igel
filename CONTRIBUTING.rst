.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at  https://github.com/nidhaloff/igel/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

igel could always use more documentation, whether as part of the
official igel docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nidhaloff/igel/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `igel` for local development.

1. Fork the `igel` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:nidhaloff/igel.git

3. Install your local copy into a virtualenv. Assuming you have poetry (https://pypi.org/project/poetry/) installed, this is how you set up your fork for local development::

    $ cd igel/
    $ poetry shell
    $ poetry update
    $ poetry install

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ make test

   To get poetry, just run pip install poetry.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.

## Releasing to PyPI

To release a new version:
1. Bump the version in `pyproject.toml`.
2. Commit and push your changes.
3. Create a new tag (e.g., `git tag v1.2.3 && git push origin v1.2.3`).
4. The GitHub Actions workflow will automatically build and publish the package to PyPI.

## 👋 For First-Time Contributors

Welcome! We're excited to have you contribute to igel. Here are some tips to help you get started:

- **Read the README and existing documentation** to understand the project's purpose and structure.
- **Look for issues labeled [`good first issue`](https://github.com/nidhaloff/igel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** — these are great entry points for new contributors.
- **Ask questions!** If you're unsure about anything, open an issue or comment on an existing one.
- **Fork the repository** and create a new branch for your changes.
- **Follow the code style guidelines** (see below).
- **Write clear commit messages** and link your pull request to the relevant issue (e.g., `Closes #143`).
- **Be respectful and collaborative** — we value every contribution!

### Useful Resources

- [GitHub's guide to contributing to open source](https://opensource.guide/how-to-contribute/)
- [How to create a Pull Request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
- [Semantic commit messages](https://www.conventionalcommits.org/en/v1.0.0/)

Thank you for helping make igel better!
