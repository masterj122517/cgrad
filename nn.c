#include "nn.h"
#include "engine.h"
#include <stdlib.h>

void neuron_zero_grad(Neuron* self)
{
  Value** parms = self->base.parmeters(self);
  int n = self->nin + 1;
  for (int i = 0; i < n; i++) {
    parms[i]->grad = 0;
  }
  free(parms);
}

Value** neuron_parm_fun(void* self)
{
  Neuron* n = (Neuron*)self;
  Value** parms = malloc(sizeof(Value*) * (n->nin + 1));
  for (int i = 0; i < n->nin; i++) {
    parms[i] = n->weights[i];
  }
  parms[n->nin] = n->bias;
  return parms;
}

Neuron* create_neuron(int nin, int nonlin)
{
  Neuron* self = malloc(sizeof(Neuron));
  self->weights = malloc(sizeof(Value*) * nin);
  self->nin = nin;
  for (int i = 0; i < nin; i++) {
    double random = ((double)rand() / (double)RAND_MAX) * 2.0f - 1.0f;
    self->weights[i] = value_new(random);
  }
  self->bias = value_new(0);
  self->nonlin = nonlin;
  self->base.parmeters = &neuron_parm_fun;
  return self;
}

Value* call_neuron(Neuron* self, Value** x)
{
  Value* sum = value_new(0);
  for (int i = 0; i < self->nin; i++) {
    Value* prod = valueMul(self->weights[i], x[i]);
    sum = valueAdd(sum, prod); // FIX: we got a memory issue here to be fixed
  }
  Value* res = valueAdd(sum, self->bias);

  return self->nonlin ? relu(res) : res;
}

void free_neuron(Neuron* n)
{
  if (!n)
    return;
  for (int i = 0; i < n->nin; i++) {
    free_value(n->weights[i]); // 数组里的Value
  }
  free(n->bias);
  free(n->weights); // 数组本身
  free(n); // Neuron本身
}

Value** layer_parm_fun(void* self)
{
  Layer* layer = (Layer*)self;
  int parms_per_neuron = layer->nin + 1;
  int total_parms = parms_per_neuron * layer->nout;
  Value** parms = malloc(sizeof(Value*) * total_parms);
  for (int i = 0; i < layer->nout; i++) {
    Value** n_parms = layer->neurons[i]->base.parmeters(layer->neurons[i]);
    for (int j = 0; j < parms_per_neuron; j++) {
      parms[i * parms_per_neuron + j] = n_parms[j];
    }
    free(n_parms);
  }
  return parms;
}

// TODO: rewrite this  <25-12-25, MasterJ>

Value** mlp_parm_fun(void* self)
{
  MLP* mlp = (MLP*)self;

  // 1. 第一次遍历：计算所有层参数的总和
  int total_count = 0;
  for (int i = 0; i < mlp->n_layers; i++) {
    total_count += mlp->layers[i]->nout * (mlp->layers[i]->nin + 1);
  }

  // 2. 分配最终的总账本（容器）
  Value** all_parms = malloc(sizeof(Value*) * total_count);
  if (!all_parms)
    return NULL;

  // 3. 第二次遍历：递归收集并摊平
  int current_offset = 0;
  for (int i = 0; i < mlp->n_layers; i++) {
    // 调用 Layer 级别的参数收集函数
    Value** l_parms = mlp->layers[i]->base.parmeters(mlp->layers[i]);

    // 计算这一层有多少个参数
    int n_l_parms = mlp->layers[i]->nout * (mlp->layers[i]->nin + 1);

    // 搬运指针
    for (int j = 0; j < n_l_parms; j++) {
      all_parms[current_offset++] = l_parms[j];
    }

    // ！！！管家的职业操守提醒：释放 Layer 函数产生的临时小数组
    free(l_parms);
  }

  return all_parms;
}

Layer* create_layer(int nin, int nout, int nonlin)
{
  Layer* layer = malloc(sizeof(Layer));
  layer->nin = nin;
  layer->nout = nout;
  layer->neurons = malloc(sizeof(Neuron*) * nout);
  for (int i = 0; i < nout; i++) {
    layer->neurons[i] = create_neuron(nin, nonlin);
  }
  layer->base.parmeters = &layer_parm_fun;
  return layer;
}

void free_layer(Layer* layer)
{
  if (!layer)
    return;
  for (int i = 0; i < layer->nout; i++) {
    free_neuron(layer->neurons[i]);
  }
  free(layer->neurons);
  free(layer);
}
Value** call_layer(Layer* layer, Value** x)
{
  Value** out = malloc((sizeof(Neuron*) * layer->nout));
  if (!out)
    return NULL;
  for (int i = 0; i < layer->nout; i++) {
    out[i] = call_neuron(layer->neurons[i], x);
  }
  return out;
}

MLP* create_mlp(int nin, int* nouts, int n_layers)
{
  MLP* self = malloc(sizeof(MLP));
  self->n_layers = n_layers;
  self->layers = malloc(sizeof(Layer*) * n_layers);
  int current_nin = nin;
  for (int i = 0; i < n_layers; i++) {
    int is_last = (i == n_layers - 1);
    self->layers[i] = create_layer(current_nin, nouts[i], !is_last);
    current_nin = nouts[i];
  }
  self->base.parmeters = &mlp_parm_fun;
  return self;
}

Value** call_mlp(MLP* self, Value** x)
{
  for (int i = 0; i < self->n_layers; i++) {
    x = call_layer(self->layers[i], x);
  }
  return x;
}

void free_mlp(MLP* mlp)
{
  if (!mlp)
    return;

  // 1. 先释放每一层 (Layers)
  for (int i = 0; i < mlp->n_layers; i++) {
    // 调用我们之前写好的 free_layer
    free_layer(mlp->layers[i]);
  }

  // 2. 释放存放 Layer 指针的数组本身
  free(mlp->layers);

  // 3. 最后释放 MLP 结构体本体
  free(mlp);
}
