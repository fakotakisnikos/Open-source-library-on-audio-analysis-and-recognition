from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='tabtools',
    version='0.1.2',
    description='Machine learning methods to work with audio inhalation analysis and recognition, in Python',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='VVR_Group',
    packages=find_packages(),
    author='Nikos D. Fakotakis',
    author_email='fakotakisnikos@gmail.com',
    keywords=['Resiratory', 'Inhalers', 'AudioClassification'],
    url='https://github.com/ncthuc/elastictools',
    download_url='https://pypi.org/project/elastictools/'
)

install_requires = [
    'elasticsearch>=6.0.0,<7.0.0',
    'jinja2'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
    setup(include_package_data = True)