from setuptools import setup, find_packages

setup(
    name='openpifpaf_wholebody',
    packages= ['openpifpaf_wholebody'],
    license='GNU AGPLv3',
    version = '0.1.0',
    description='OpenPifPaf wholebody Extension',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Duncan Zauss',
    url='https://github.com/vita-epfl/openpifpaf_wholebody',

    install_requires=[
        'matplotlib',
        'openpifpaf>=0.12b1',
    ],
)
