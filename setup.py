import setuptools

with open('README.md','r',encoding="utf-8") as f:
    long_description=f.read()




__version__= "0.0.0"

REPO_NAME='ProjetBdMlops'
AUTHOR_USER_NAME="RaffouIUT"
SRC_REPO = "ProjetBdMlops"
AUTHOR_EMAIL="Rafael.Doneau.Etu@univ-lemans.fr"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='Projet MLOps ',
    long_description=long_description,
    long_destription_content='text/markdown',
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={

        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src'),


)
