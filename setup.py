from setuptools import setup, find_packages

setup(
    name='openpifpaf_wholebody',
    packages= ['openpifpaf_wholebody'],
    version = '0.1.0',
    description='OpenPifPaf wholebody',
    author='Duncan Zauss',
    url='https://github.com/vita-epfl/openpifpaf_wholebody',

    install_requires=[
        'matplotlib',
        'openpifpaf>=0.12b1',
    ],
)
