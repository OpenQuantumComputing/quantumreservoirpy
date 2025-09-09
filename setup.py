from setuptools import setup

INSTALL_REQUIRES = [
    "matplotlib",
    "tqdm",
    "pylatexenc>=2.0",
    "qiskit<2.1.0",
    "qiskit-aer>=0.12.0",
    "numpy>=1.17",
    "scikit-learn",
    "ipykernel"
    "stim"
    "bitarray"
]

setup(
    name='quantumreservoirpy',
    version='0.2',
    packages=['quantumreservoirpy'],
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.9',
)
