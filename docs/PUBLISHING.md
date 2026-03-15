# How to Make This Accessible

The goal of this project is to be an open-source tool available to any actuary in the world. 

To achieve this, the project has been packaged using `scikit-build-core`, meaning the complex C++ CMake configuration is completely hidden from the end user. They simply need to type `pip install open-riskpy`.

## 1. Publishing to GitHub
The first step to accessibility is open-sourcing the code so the community can contribute new Actuarial C++ models.
```bash
git init
git add .
git commit -m "Initial Release of Actuarial Underwriter"
git branch -M main
git remote add origin https://github.com/slimboi34/RiskPY.git
git push -u origin main
```

## 2. Publishing to PyPI (Python Package Index)
To allow anyone to globally install your library via `pip`, you must upload the package to PyPI. 

First, install the Python build and upload tools:
```bash
pip install build twine
```

Next, generate the distribution archives (this will create `.tar.gz` and `.whl` files in a `dist/` folder):
```bash
python -m build
```

Finally, upload the distributions to PyPI. To do this, you must first create an account and generate an API token explicitly for `twine`:

1. **Sign up for an account**: Navigate to [PyPI Registration](https://pypi.org/account/register/) to create your free account.
2. **Setup Two-Factor Authentication**: PyPI requires 2FA to publish packages (using a mobile authenticator app or security key).
3. **Generate an API Token**: 
   - Go to your [PyPI Account Settings](https://pypi.org/manage/account/).
   - Scroll down to the **API tokens** section and click **Add API token**.
   - Input a token name (e.g., `riskpy-publisher`) and select "Entire account" for scope.
   - *Save the generated token securely* (it starts with `pypi-`).

4. **Upload the Package**: Run the following command in your terminal. When it prompts for a username, type `__token__`. When it prompts for the password, paste your `pypi-` token.
```bash
twine upload dist/*
```

## 3. User Installation
Once published to PyPI, any actuary, data scientist, or underwriter can install the entire C++ framework on their machine by running:
```bash
pip install open-riskpy
```
*(Note: Because this is a C++ Pybind extension, the user must have a C++ compiler installed on their system, such as GCC, Clang, or MSVC, which is standard on most developer machines).*
