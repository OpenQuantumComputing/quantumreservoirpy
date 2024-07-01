from setuptools import setup

INSTALL_REQUIRES = [
    "matplotlib",
    "pylatexenc>=2.0",
    "qiskit>=1.0",
    "qiskit-aer>=0.12.0",
    "numpy>=1.21.6",
    "scikit-learn",
    "ipykernel"
]

setup(name='quantumreservoirpy',
      version='0.2',
      packages=['quantumreservoirpy'],
      install_requires=INSTALL_REQUIRES)
