# chessmait

![](documentation/logo.jpg)

# Set up and handle environment

After cloning the repository, you can create a virtual environment with the necessary dependencies like his:

```shell
$ pipenv install --dev
```

Install a new dependency:

```shell
$ pipenv install <library>
```

Update `Pipfile.lock` with the current state of the virtual environment:

```shell
$ pipenv lock
```

# Execute the tests

To execute the tests, you can simply call pytest in the root directory (if you get an error regarding the modules, set
the variable `$PYTHONPATH` to the project's root path):

```shell
$ pytest
```



