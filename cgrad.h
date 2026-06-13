// cgrad.h

#ifndef CGRAD_H
#define CGRAD_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Tensor Tensor;

// 创建
Tensor* cg_tensor(float* data, int* shape, int ndim);

// 运算
Tensor* cg_add(Tensor* a, Tensor* b);
Tensor* cg_matmul(Tensor* a, Tensor* b);
Tensor* cg_sum(Tensor* a);
Tensor* cg_relu(Tensor* a);
Tensor* cg_softmax_cross_entropy(Tensor* logits, Tensor* y_true);
Tensor* cg_randn(int* shape, int ndim, float scale);
Tensor* cg_zeros(int* shape, int ndim);
void cg_sgd_step(Tensor* t, float lr);
void cg_zero_grad(Tensor* t);

// 反向传播
void cg_backward(Tensor* t);

// 工具
void cg_print(Tensor* t);
void cg_free(Tensor* t);

// internal — 不给用户用，但需要 forward declare
void backward_add(Tensor* t);
void backward_matmul(Tensor* t);
void backward_sum(Tensor* t);
void backward_relu(Tensor* t);
void backward_softmax_cross_entropy(Tensor* t);

#endif // CGRAD_H

#ifdef CGRAD_IMPLEMENTATION

// ---- IMPLEMENTATION ----
struct Tensor
{
  float* data;
  float* grad;
  int* shape;
  int ndim;
  // backward函数指针
  void (*_backward)(struct Tensor*);
  struct Tensor* children[2];
};

typedef struct
{
  Tensor** data;
  int size;
  int capacity;
} TensorVec;

TensorVec vec_new(int initial_capacity)
{
  return (TensorVec){
      .data = malloc(initial_capacity * sizeof(Tensor*)),
      .size = 0,
      .capacity = initial_capacity,
  };
}

void vec_push(TensorVec* v, Tensor* t)
{
  if (v->size >= v->capacity) {
    v->capacity *= 2;
    v->data = realloc(v->data, v->capacity * sizeof(Tensor*));
  }
  v->data[v->size++] = t;
}

int vec_contains(TensorVec* v, Tensor* t)
{
  for (int i = 0; i < v->size; i++)
    if (v->data[i] == t)
      return 1;
  return 0;
}

void vec_free(TensorVec* v)
{
  free(v->data);
  v->data = NULL;
  v->size = v->capacity = 0;
}

// internal: takes ownership of data (data must be heap-allocated)
static Tensor* cg__tensor_owned(float* data, int* shape, int ndim)
{
  Tensor* tensor = malloc(sizeof(Tensor));
  if (tensor == NULL)
    return NULL;
  tensor->data = data;
  tensor->ndim = ndim;
  tensor->shape = malloc(ndim * sizeof(int));
  memcpy(tensor->shape, shape, ndim * sizeof(int));
  int size = 1;
  for (int i = 0; i < ndim; i++)
    size *= shape[i];
  tensor->grad = calloc(size, sizeof(float));

  tensor->children[0] = NULL;
  tensor->children[1] = NULL;
  tensor->_backward = NULL;

  return tensor;
}

Tensor* cg_tensor(float* data, int* shape, int ndim)
{
  int size = 1;
  for (int i = 0; i < ndim; i++)
    size *= shape[i];
  float* copy = malloc(size * sizeof(float));
  memcpy(copy, data, size * sizeof(float));
  return cg__tensor_owned(copy, shape, ndim);
}

Tensor* cg_add(Tensor* a, Tensor* b)
{
  if ((a->ndim != b->ndim))
    return NULL;

  for (int i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i])
      return NULL;
  }

  int size = 1;
  for (int i = 0; i < a->ndim; i++) {
    size *= a->shape[i];
  }
  float* result_data = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    result_data[i] = a->data[i] + b->data[i];
  }

  Tensor* result = cg__tensor_owned(result_data, a->shape, a->ndim);
  result->children[0] = a;
  result->children[1] = b;
  result->_backward = backward_add;
  return result;
}

void backward_add(Tensor* t)
{
  Tensor* a = t->children[0];
  Tensor* b = t->children[1];
  int size = 1;
  for (int i = 0; i < t->ndim; i++)
    size *= t->shape[i];
  for (int i = 0; i < size; i++) {
    a->grad[i] += t->grad[i];
    b->grad[i] += t->grad[i];
  }
}

void build_topo(Tensor* node, TensorVec* topo, TensorVec* visited)
{
  if (vec_contains(visited, node))
    return;
  vec_push(visited, node);

  if (node->children[0])
    build_topo(node->children[0], topo, visited);
  if (node->children[1])
    build_topo(node->children[1], topo, visited);

  vec_push(topo, node);
}

void cg_backward(Tensor* t)
{
  // 只能对标量调用
  assert(t->ndim == 1 && t->shape[0] == 1);

  TensorVec topo = vec_new(8);
  TensorVec visited = vec_new(8);

  // build the dag graph
  build_topo(t, &topo, &visited);

  // back the dL/dL = 1 where the chain rule starts
  t->grad[0] = 1.0f;

  // 反向调用topo，传播梯度
  for (int i = topo.size - 1; i >= 0; i--) {
    if (topo.data[i]->_backward)
      topo.data[i]->_backward(topo.data[i]);
  }

  vec_free(&topo);
  vec_free(&visited);
}

// (M, N) -> (N, M)
float* transpose(float* data, int rows, int cols)
{
  float* result = malloc(rows * cols * sizeof(float));
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result[j * rows + i] = data[i * cols + j];
  return result;
}

// matmul helper: C = A @ B, shapes: (M,N) @ (N,P) -> (M,P)
void matmul_into(float* A, float* B, float* C, int M, int N, int P)
{
  memset(C, 0, M * P * sizeof(float));
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++)
      for (int k = 0; k < N; k++)
        C[i * P + j] += A[i * N + k] * B[k * P + j];
}

void backward_matmul(Tensor* t)
{
  Tensor* a = t->children[0]; // (M, N)
  Tensor* b = t->children[1]; // (N, P)

  int M = a->shape[0];
  int N = a->shape[1];
  int P = b->shape[1];

  // dL/dA = dL/dC @ B^T   (M,P) @ (P,N) -> (M,N)
  float* bt = transpose(b->data, N, P); // (P, N)
  float* dA = malloc(M * N * sizeof(float));
  matmul_into(t->grad, bt, dA, M, P, N);
  for (int i = 0; i < M * N; i++)
    a->grad[i] += dA[i];

  // dL/dB = A^T @ dL/dC   (N,M) @ (M,P) -> (N,P)
  float* at = transpose(a->data, M, N); // (N, M)
  float* dB = malloc(N * P * sizeof(float));
  matmul_into(at, t->grad, dB, N, M, P);
  for (int i = 0; i < N * P; i++)
    b->grad[i] += dB[i];

  free(bt);
  free(at);
  free(dA);
  free(dB);
}

// index = i * C + j (i,j)
Tensor* cg_matmul(Tensor* a, Tensor* b)
{
  if (a->ndim != 2 || b->ndim != 2)
    return NULL;
  if (a->shape[1] != b->shape[0])
    return NULL;

  int M = a->shape[0];
  int N = a->shape[1];
  int P = b->shape[1];

  int res_shape[2] = {M, P};
  float* res_buffer = calloc(M * P, sizeof(float));

  matmul_into(a->data, b->data, res_buffer, M, N, P);

  Tensor* result = cg__tensor_owned(res_buffer, res_shape, 2);
  result->children[0] = a;
  result->children[1] = b;
  result->_backward = backward_matmul;
  return result;
}

void backward_sum(Tensor* t)
{
  Tensor* a = t->children[0];
  int size = 1;
  for (int i = 0; i < a->ndim; i++)
    size *= a->shape[i];

  // sum的梯度就是把上游梯度广播到所有元素
  for (int i = 0; i < size; i++)
    a->grad[i] += t->grad[0];
}

Tensor* cg_sum(Tensor* a)
{
  int size = 1;
  for (int i = 0; i < a->ndim; i++)
    size *= a->shape[i];

  float* res_data = malloc(sizeof(float));
  res_data[0] = 0.0f;
  for (int i = 0; i < size; i++)
    res_data[0] += a->data[i];

  int res_shape[1] = {1};
  Tensor* result = cg__tensor_owned(res_data, res_shape, 1);
  result->children[0] = a;
  result->children[1] = NULL;
  result->_backward = backward_sum;
  return result;
}

void cg_print(Tensor* t)
{
  printf("Tensor shape=(");
  for (int i = 0; i < t->ndim; i++) {
    printf("%d", t->shape[i]);
    if (i < t->ndim - 1)
      printf(", ");
  }
  printf(")\n");

  int size = 1;
  for (int i = 0; i < t->ndim; i++)
    size *= t->shape[i];

  printf("data: ");
  for (int i = 0; i < size; i++)
    printf("%.4f ", t->data[i]);
  printf("\n");

  printf("grad: ");
  for (int i = 0; i < size; i++)
    printf("%.4f ", t->grad[i]);
  printf("\n");
}

void cg_free(Tensor* t)
{
  if (t == NULL)
    return;
  free(t->data);
  free(t->grad);
  free(t->shape);
  free(t);
}

void backward_relu(Tensor* t)
{
  Tensor* a = t->children[0];
  int size = 1;
  for (int i = 0; i < a->ndim; i++) {
    size *= a->shape[i];
  }
  for (int i = 0; i < size; i++) {
    a->grad[i] += t->data[i] > 0 ? t->grad[i] : 0.0f;
  }
}

Tensor* cg_relu(Tensor* a)
{
  int size = 1;
  for (int i = 0; i < a->ndim; i++)
    size *= a->shape[i];

  float* res_data = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
    res_data[i] = a->data[i] > 0 ? a->data[i] : 0.0f;
  Tensor* result = cg__tensor_owned(res_data, a->shape, a->ndim);
  result->children[0] = a;
  result->children[1] = NULL;
  result->_backward = backward_relu;
  return result;
}

void backward_softmax_cross_entropy(Tensor* t)
{
  Tensor* logits = t->children[0];
  Tensor* y_true = t->children[1];
  int N;
  if (logits->ndim == 2 && logits->shape[0] == 1)
    N = logits->shape[1];
  else
    N = logits->shape[0];

  for (int i = 0; i < N; i++)
    logits->grad[i] += t->grad[0] * (t->data[i + 1] - y_true->data[i]);
}

Tensor* cg_softmax_cross_entropy(Tensor* logits, Tensor* y_true)
{
  int N = 0;
  if (logits->ndim == 2 && logits->shape[0] == 1)
    N = logits->shape[1];
  else if (logits->ndim == 1)
    N = logits->shape[0];
  else
    return NULL;

  if (y_true->shape[0] != N)
    return NULL;

  float max_val = logits->data[0];
  for (int i = 1; i < N; i++)
    if (logits->data[i] > max_val)
      max_val = logits->data[i];

  float* softmax_out = malloc(N * sizeof(float));
  float sum_exp = 0.0f;
  for (int i = 0; i < N; i++) {
    softmax_out[i] = expf(logits->data[i] - max_val);
    sum_exp += softmax_out[i];
  }
  for (int i = 0; i < N; i++)
    softmax_out[i] /= sum_exp;

  float loss_val = 0.0f;
  for (int i = 0; i < N; i++)
    loss_val -= y_true->data[i] * logf(softmax_out[i] + 1e-7f);

  // res_data[0] = loss, res_data[1..N] = softmax
  float* res_data = malloc((N + 1) * sizeof(float));
  res_data[0] = loss_val;
  memcpy(res_data + 1, softmax_out, N * sizeof(float));
  free(softmax_out);

  int res_shape[1] = {1};
  Tensor* result = cg__tensor_owned(res_data, res_shape, 1);
  result->children[0] = logits;
  result->children[1] = y_true;
  result->_backward = backward_softmax_cross_entropy;
  return result;
}

Tensor* cg_randn(int* shape, int ndim, float scale)
{
  int size = 1;
  for (int i = 0; i < ndim; i++)
    size *= shape[i];

  float* data = malloc(size * sizeof(float));

  // Box-Muller 生成正态分布
  for (int i = 0; i < size; i += 2) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(2.0f * M_PI * u2);
    float z1 = sqrtf(-2.0f * logf(u1 + 1e-7f)) * sinf(2.0f * M_PI * u2);
    data[i] = z0 * scale;
    if (i + 1 < size)
      data[i + 1] = z1 * scale;
  }

  return cg__tensor_owned(data, shape, ndim);
}

void cg_sgd_step(Tensor* t, float lr)
{
  int size = 1;
  for (int i = 0; i < t->ndim; i++)
    size *= t->shape[i];

  for (int i = 0; i < size; i++)
    t->data[i] -= lr * t->grad[i];
}

void cg_zero_grad(Tensor* t)
{
  int size = 1;
  for (int i = 0; i < t->ndim; i++)
    size *= t->shape[i];

  memset(t->grad, 0, size * sizeof(float));
}

// 实现
Tensor* cg_zeros(int* shape, int ndim)
{
  int size = 1;
  for (int i = 0; i < ndim; i++)
    size *= shape[i];
  float* data = calloc(size, sizeof(float));
  return cg__tensor_owned(data, shape, ndim);
}

// we haven't support CUDA yet.
// #ifdef CGRAD_CUDA
// ...
// #endif

#endif // CGRAD_IMPLEMENTATION
