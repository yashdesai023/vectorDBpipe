# How to Publish to PyPI

This document describes how to publish the `vectorDBpipe` package to PyPI.

## 1. Create a PyPI Account

If you don't have a PyPI account, create one at [pypi.org](https://pypi.org/account/register/).

## 2. Generate a PyPI API Token

1. Go to your PyPI account settings and select "API tokens".
2. Create a new token. Give it a name, for example, `vectorDBpipe-github-actions`.
3. **Important:** Copy the token. You will not be able to see it again.

## 3. Add the Token to GitHub Secrets

1. In your GitHub repository, go to "Settings" > "Secrets and variables" > "Actions".
2. Click "New repository secret".
3. Name the secret `PYPI_API_TOKEN`.
4. Paste the PyPI API token you copied earlier into the "Value" field.

## 4. Create a New Release on GitHub

1. Go to the "Releases" page in your GitHub repository.
2. Click "Draft a new release".
3. Create a new tag for the release, for example, `v0.1.0`.
4. Give the release a title and description.
5. Click "Publish release".

This will trigger the `publish_python.yml` workflow, which will build and publish the package to PyPI.
