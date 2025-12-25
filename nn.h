
#ifndef NN_H
#define NN_H

#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*zero_grad)(void* self);
typedef Value** (*get_parmeters)(void* self);

typedef struct
{
  zero_grad zero_grad;
  get_parmeters parmeters;
} Module;

typedef struct
{
  Module base;
  Value** weights;
  Value* bias;
  int nin;
  int nonlin;

} Neuron;

typedef struct
{
  Module base;
  Neuron** neurons;
  int nout;
  int nin;
} Layer;

typedef struct
{
  Module base;
  Layer** layers;
  int n_layers;

} MLP;

void free_neuron(Neuron* n);

Neuron* create_neuron(int nin, int nonlin);

Value* call_neuron(Neuron* self, Value** x);

Layer* create_layer(int nin, int nout, int nonlin);

void free_layer(Layer* layer);

Value** call_layer(Layer* layer, Value** x);

MLP* create_mlp(int nin, int* nouts, int n_layers);

Value** call_mlp(MLP* self, Value** x);

void free_mlp(MLP* mlp);

#endif // !NN_H
