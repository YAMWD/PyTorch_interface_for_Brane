# PyTorch interface for Brane
This project features an implementation of an interface for some Pytorch functions using the [Brane](https://github.com/epi-project/brane) framework. The implementation is composed of some brane scripts to call the corresponding functions and a package with all the source files in it.

# Build 
To build the brane package, user can nevigate to the package/torch directory and run the following command.
```
cd package/torch
brane build container.yml
```

# Run
User can excute out implementation locally by simply running the following command in the root folder of the project:

```bash
brane run <scriptname>.bs
```

