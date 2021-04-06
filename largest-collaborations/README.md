# largest-collaborations

Create the table `project_size_tabular.tex` (appears in Figure 1 of the paper).

## Install

```
pipenv sync
```

## Usage

```
pipenv run python main.py load analyze
```

## Methodology

The script works in two steps, first it downloads statistics about repos on GitHub and then
it analyzes the local data to create the table.
