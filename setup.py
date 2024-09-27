from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e.'


def get_requirements(path:str)->List[str]:
    try:
        requirements = []
        with open ("file","r") as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements

    except Exception as e:
        print(e)

setup(
    name="mashroom classification",
    version="0.1",
    description="project on mushroom classification",
    author="ankit",
    author_email="ankitsharma450306@gmail.com",
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages(),
    


)