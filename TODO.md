# cgrad — TODO

> **How little runtime do you need to train a neural network well?**

Not decided yet. That's the point.

## What I know

I want to rewrite `cgrad`.

Not just because the old version needs more features, and not to build another tiny PyTorch.

The interesting question is:

**How small can the machinery required for neural network training be, while still achieving respectable performance?**

"Small" does **not** just mean fewer lines of code.

I care about the tradeoff between:

- simplicity
- runtime complexity
- dependencies
- binary/runtime size
- understandability
- actual training performance

The goal is not to beat PyTorch or tinygrad.

The interesting result would be:

> *Wait, this little runtime can actually train that? And it's reasonably fast?*

---

## Current idea of the runtime

Maybe the runtime only needs to provide machinery for:

```text
numerical data
      ↓
numerical operations
      ↓
automatic differentiation
      ↓
memory management
      ↓
efficient hardware execution
```

Possibly:

```text
cgrad
├── Tensor / Storage
├── Ops
├── Autograd
├── Memory
└── Kernels
```

Things like:

```text
Linear
Attention
Transformer
SGD / Adam
Trainer
Dataset
Module
```

may not belong in the runtime at all.

If they can be efficiently expressed using lower-level primitives, they should probably remain user-land code.

But none of this is decided yet.

---

## Important realization

Don't start by copying PyTorch or tinygrad architecture.

Don't assume we need:

```text
IR
graph
lazy execution
compiler
VM
JIT
GPU backend
```

If a benchmark eventually gives us a reason to introduce one of them, then it has earned its place.

The architecture should emerge from the workload.

---

## Possible first workloads

Start stupidly small:

```text
XOR / tiny MLP
      ↓
MLP training benchmark
      ↓
character language model
      ↓
tiny Transformer
```

The original fun idea is still useful:

> train a tiny fake chat model entirely using cgrad.

Not because the model matters.

Because it would be funny if a tiny C runtime could actually train something that talks.

---

## Performance matters

A 1,000-line runtime that is 100× slower isn't necessarily the answer.

The interesting region is:

```text
              Performance
                   ↑
                   │
                   │      huge frameworks
                   │
                   │   ★ cgrad?
                   │
                   │
                   │ × tiny toy engine
                   │
                   └──────────────────→ Complexity
```

Push ★ toward the upper-left.

Maybe useful measurements are:

```text
runtime LOC
dependencies
binary size

matmul performance
MLP training speed
char-LM training speed
tiny Transformer training speed
memory usage
```

No benchmark bullshit.

The question isn't:

> "Can cgrad beat PyTorch?"

It's:

> **"How much performance can we buy with very little machinery?"**

---

# Tomorrow

Don't code immediately.

First answer:

### 1. What exactly counts as the runtime?

Draw the boundary.

### 2. What does "small" mean?

Pick measurable constraints instead of vibes.

### 3. What does "good performance" mean?

Choose references and acceptable performance targets.

### 4. What is the first benchmark?

Probably:

```text
matmul microbenchmark
+
small MLP training benchmark
```

### 5. What is the first experiment?

Start with almost nothing.

Add machinery only when correctness, capability, or benchmarks force us to.

---

## One rule

Before adding an abstraction, ask:

> **What problem did we actually encounter that requires this?**

If there isn't a concrete answer, don't add it.

---

## The question to come back to

**How little runtime do you need to train a neural network well?**

Tomorrow, figure out what that sentence actually means.
