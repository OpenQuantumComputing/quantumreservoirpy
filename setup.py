from setuptools import setup

INSTALL_REQUIRES = [
    "matplotlib>=3.7.1",
    "pylatexenc>=2.0",
    "qiskit>=0.43.1",
    "qiskit-aer>=0.12.0",
    "numpy>=1.23.5",
    "scikit-learn",
    "ipykernel"
]

setup(name='quantumreservoirpy',
      version='0.1',
      packages=['quantumreservoirpy'],
      install_requires=INSTALL_REQUIRES)
