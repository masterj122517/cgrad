// implementation for Values and calculations
#include "engine.h"
#include <math.h>
#include <stdlib.h>

// init value
Value* value_new(double x)
{
  Value* v = malloc(sizeof(Value));
  v->data = x;
  v->grad = 0;
  v->children = NULL;
  v->n_children = 0;
  v->ops = 0;
  v->powN = 0;
  v->backward = NULL;
  return v;
}

int is_visited(Value* v, Value** visited, int visited_size)
{
  for (int i = 0; i < visited_size; i++) {
    if (visited[i] == v)
      return 1;
  }
  return 0;
}

static void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size)
{
  if (is_visited(v, visited, *visited_size))
    return;

  visited[(*visited_size)++] = v;
  for (int i = 0; i < v->n_children; i++) {
    build_topo(v->children[i], topo, topo_size, visited, visited_size);
  }
  topo[(*topo_size)++] = v;
}

void backward(Value* root)
{
  Value* topo[1024];
  Value* visited[1024];
  int topo_size = 0;
  int visited_size = 0;

  build_topo(root, topo, &topo_size, visited, &visited_size);

  root->grad = 1.0;

  for (int i = topo_size - 1; i >= 0; i--) {
    if (topo[i]->backward) {
      topo[i]->backward(topo[i]);
    }
  }
}

static void addBackward(Value* out)
{
  out->children[0]->grad += out->grad;
  out->children[1]->grad += out->grad;
}
static void addScalarBackward(Value* out)
{
  Value* a = out->children[0];
  a->grad += out->grad;
}

static void subBackward(Value* out)
{
  out->children[0]->grad += out->grad; // ∂(a-b)/∂a = 1
  out->children[1]->grad -= out->grad; // ∂(a-b)/∂b = -1  注意这里是 -=
}

static void mulBackward(Value* out)
{
  Value* a = out->children[0];
  Value* b = out->children[1];

  a->grad += b->data * out->grad;
  b->grad += a->data * out->grad;
}
static void mulScalarBackward(Value* out)
{
  Value* a = out->children[0];
  double b = out->powN;
  a->grad += b * out->grad;
}

static void powBackward(Value* out)
{
  out->children[0]->grad += out->grad * pow(out->children[0]->data, out->powN - 1) * out->powN;
}

static void divBackward(Value* out)
{
  Value* a = out->children[0];
  Value* b = out->children[1];

  a->grad += (1.0 / b->data) * out->grad;
  b->grad += (-a->data / (b->data * b->data)) * out->grad;
}

static void scalarDivBackward(Value* out)
{
  Value* a = out->children[0]; // scalar wrapper
  Value* b = out->children[1];

  // ∂(a / b)/∂b = -a / b^2
  b->grad += (-a->data / (b->data * b->data)) * out->grad;
}

static void reluBackward(Value* out)
{
  Value* a = out->children[0];
  double grad = (out->data > 0) ? 1.0 : 0.0;
  a->grad += grad * out->grad;
}

Value* valueAdd(Value* a, Value* b)
{
  double res = a->data + b->data;
  Value* out = value_new(res);
  out->n_children = 2;
  out->children = malloc(sizeof(Value*) * 2);
  out->children[0] = a;
  out->children[1] = b;
  out->ops = "+";
  out->backward = &addBackward;
  return out;
}
Value* valueAddScalar(Value* a, double b)
{
  double res = a->data + b;
  Value* out = value_new(res);
  out->n_children = 1;
  out->children = malloc(sizeof(Value*));
  out->children[0] = a;
  out->ops = "+";
  out->backward = &addScalarBackward;
  return out;
}

Value* valueMul(Value* a, Value* b)
{

  double res = a->data * b->data;
  Value* out = value_new(res);
  out->n_children = 2;
  out->children = malloc(sizeof(Value*) * 2);
  out->children[0] = a;
  out->children[1] = b;
  out->ops = "*";
  out->backward = &mulBackward;
  return out;
}

Value* valueMulScalar(Value* a, double b)
{
  double res = a->data * b;
  Value* out = value_new(res);
  out->n_children = 1;
  out->children = malloc(sizeof(Value*));
  out->children[0] = a;
  out->ops = "*";
  out->backward = &mulScalarBackward;
  out->powN = b; // remember it is not pow is b's value
  return out;
}

Value* valuePow(Value* a, int n)
{
  double res = pow(a->data, n);
  Value* out = value_new(res);

  out->n_children = 1;
  out->children = malloc(sizeof(Value*));
  out->children[0] = a;

  out->ops = "^";
  out->powN = n;
  out->backward = &powBackward;

  return out;
}
Value* relu(Value* a)
{
  Value* out = value_new(a->data > 0 ? a->data : 0.0);

  out->n_children = 1;
  out->children = malloc(sizeof(Value*));
  out->children[0] = a;

  out->ops = "relu";
  out->backward = &reluBackward;

  return out;
}

Value* valueNeg(Value* a)
{
  double neg = -1.0;
  return valueMulScalar(a, neg);
}
Value* valueSub(Value* a, Value* b)
{
  Value* out = value_new(a->data - b->data);

  out->n_children = 2;
  out->children = malloc(sizeof(Value*) * 2);
  out->children[0] = a;
  out->children[1] = b;

  out->ops = "-";
  out->backward = &subBackward;

  return out;
}
Value* valueSubScalar(Value* a, double b)
{
  return valueAddScalar(a, -b);
}

Value* valueTruediv(Value* a, Value* b)
{
  Value* out = value_new(a->data / b->data);

  out->n_children = 2;
  out->children = malloc(2 * sizeof(Value*));
  out->children[0] = a;
  out->children[1] = b;

  out->ops = "/";
  out->backward = &divBackward;

  return out;
}
Value* valueTruedivScalar(Value* a, double b)
{
  double divB = pow(b, -1);
  return valueMulScalar(a, divB);
}

Value* valueScalarTruediv(double a, Value* b)
{
  Value* av = value_new(a);
  Value* out = value_new(a / b->data);

  out->n_children = 2;
  out->children = malloc(2 * sizeof(Value*));
  out->children[0] = av;
  out->children[1] = b;

  out->ops = "/scalar";
  out->backward = &scalarDivBackward;

  return out;
}

void free_value(Value* v)
{
  if (!v)
    return;
  if (v->children != NULL) {
    free(v->children);
  }
  free(v);
}
