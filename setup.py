import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transfer-sharmaparth",
    version="0.1.0",
    author="Parth Sharma",
    author_email="sharmaparth17@gmail.com",
    description="Python package for transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parth1993/transfer",
    packages=setuptools.find_packages(),
    classifiers=[
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: C',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     ('Programming Language :: Python :: '
                      'Implementation :: CPython'),
                     ('Programming Language :: Python :: '
                      'Implementation :: PyPy')
                     ],
                ],
    python_requires='>=3.6',
)