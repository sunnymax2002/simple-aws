# How to Build and Publish Package on PyPI

Following instructions are extracted from [Real Python guide](https://realpython.com/pypi-publish-python-package/#create-a-small-python-package) on this topic.

1. Enter the root git directory

2. Install locally in edit mode (This will link from source and later installations of the package won't work)

    python -m pip install -e .

3. Build

    python -m build

4. Check that build worked correctly

    twine check dist/*

5. Also check how the package will be installed in site_packages, by copying .whl as a .zip and extracting

5. Upload to TestPyPI

    twine upload -r testpypi dist/*

6. Check where package installed

    python -m pip -v list