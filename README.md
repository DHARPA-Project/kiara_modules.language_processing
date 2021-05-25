[![PyPI status](https://img.shields.io/pypi/status/kiara_modules.language_processing.svg)](https://pypi.python.org/pypi/kiara/)
[![PyPI version](https://img.shields.io/pypi/v/kiara_modules.language_processing.svg)](https://pypi.python.org/pypi/kiara/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/kiara_modules.language_processing.svg)](https://pypi.python.org/pypi/kiara/)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FDHARPA-Project%2Fkiara%2Fbadge%3Fref%3Ddevelop&style=flat)](https://actions-badge.atrox.dev/DHARPA-Project/kiara_modules.language_processing/goto?ref=develop)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# *kiara* modules (language_processing)

A set of commonly used/useful default modules (and pipelines) for [*Kiara*](https://github.com/DHARPA-project/kiara).

 - Documentation: [https://dharpa.org/kiara_modules.language_processing](https://dharpa.org/kiara_modules.language_processing)
 - Code: [https://github.com/DHARPA-Project/kiara_modules.language_processing](https://github.com/DHARPA-Project/kiara_modules.language_processing)

## Description

TODO

## Development

### Requirements

- Python (version >=3.6 -- some make targets only work for Python >=3.7 though)
- pip, virtualenv
- git
- make
- [direnv](https://direnv.net/) (optional)


### Prepare development environment

If you only want to work on the modules, and not the core *Kiara* codebase, follow the instructions below. Otherwise, please
check the notes on how to setup a *Kiara* development environment under (TODO).

```console
git clone https://github.com/DHARPA-Project/kiara_modules.language_processing.git
cd kiara_modules.language_processing
python3 -m venv .venv
source .venv/bin/activate
make init
```

After this is done, you should be able to run the included example module via:

```console
# for now we need to install required spacy modules manually:
python -m spacy download it_core_news_sm
# then, using the data from the TopicModeling repository:
kiara run -d topic_modelling_end_to_end path=path/to/data_tm_workflow earliest="1919-01-01" latest="2000-01-01" languages=italian languages=german compute_coherence=true
...
...
```

### ``make`` targets

- ``init``: init development project (install project & dev dependencies into virtualenv, as well as pre-commit git hook)
- ``update-dependencies``: update development dependencies (mainly the core ``kiara`` package from git)
- ``flake``: run *flake8* tests
- ``mypy``: run mypy tests
- ``test``: run unit tests
- ``docs``: create static documentation pages (under ``build/site``)
- ``serve-docs``: serve documentation pages (incl. auto-reload) for getting direct feedback when working on documentation
- ``clean``: clean build directories

For details (and other, minor targets), check the ``Makefile``.


### Running tests

``` console
> make test
# or
> make coverage
```


## Copyright & license

This project is MPL v2.0 licensed, for the license text please check the [LICENSE](/LICENSE) file in this repository.

[Copyright (c) 2021 DHARPA project](https://dharpa.org)
