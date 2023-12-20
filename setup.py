from distutils.core import setup

setup(
    name='PLEX',
    version='0.0.1',
    author='PLEX authors',
    author_email='akolobov@microsoft.com',
    packages=['PLEX'],
    url='https://github.com/microsoft/PLEX',
    license='MIT LICENSE',
    description='An implementation of the PLanning-EXecution (PLEX) architecture to accompany the paper (https://arxiv.org/abs/2303.08789)',
    long_description=open('README.md').read(),
    install_requires=[
        "attrdict==2.0.1",
        "h5py==3.4.0",
        "numpy<=1.23.5",
        "mujoco==2.1.5",
        "robosuite==1.3.2",
        "egl_probe@git+https://github.com/StanfordVL/egl_probe",
        "robomimic@git+https://github.com/akolobov/robomimic",
        "r3m@git+https://github.com/akolobov/r3m",
        "setuptools==66",
        "gym==0.20.0",
        "torch<=2.0.1",
        "torchvision",
        "transformers==4.36.0",
        "opencv-python==4.5.3.56",
        "opencv-python-headless==4.3.0.36",
        "wandb==0.9.1",
        "moviepy",
        "deepdiff",
        "metaworld@git+https://github.com/microsoft/PLEX-Metaworld"
    ]
)