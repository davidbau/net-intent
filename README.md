NetIntent
---------

A simple experiment for totaling up backprop using a jacobian method.

To use this you need python3, pip3, and python3-venv.  Of course it
runs best on a GPU machine; if you have one, you should have CUDA, CUDNN, etc.

It will download and build all other dependencies such as Theano and Blocks
in a self-contained virtual environment under the env directory.

```
  sudo apt-get install python3
  sudo apt-get install python3-pip
  sudo apt-get install python3.4-venv
```

Then simply `make`, and things should run.
