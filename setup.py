from setuptools import setup, find_packages

setup(
    name='openpifpaf_wholebody',
    packages= ['openpifpaf_wholebody'],
    version = '0.0.1',
    description='OpenPifPaf wholebody',
    author='Duncan Zauss',
    author_email='duncan.zauss@gmx.net',
    url='https://github.com/vita-epfl/openpifpaf_wholebody',

    install_requires=[
        'matplotlib',
        'openpifpaf>=0.12b1',
    ],
)
