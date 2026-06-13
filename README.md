# cgrad — a tiny autograd engine in C

> *"What if PyTorch, but make it C — and you can read the whole thing in 10 minutes?"*

**cgrad** is a single-header  tensor autograd library. It does reverse-mode automatic differentiation, element-wise ops, matrix multiply, ReLU, softmax cross-entropy — the whole gradient descent stack — in pure C. No dependencies beyond `libc` and `libm`.

And it's a toy project, no support for gpu ...  


we use [nob.h](https://github.com/tsoding/nob.h/) to build the tests

just run 

```shell
cc nob.c -o nob && ./nob
```

then have fun
