// API for Values and calculations

#ifndef ENGINE_H
#define ENGINE_H

typedef struct Value Value;

typedef void (*backward_fn)(Value* self);

typedef struct Value
{
  double data;
  double grad;
  struct Value** children;
  int n_children;
  char* ops;
  backward_fn backward;
  double powN;

} Value;

Value* value_new(double x);

Value* relu(Value* v);

Value* valueAdd(Value* a, Value* b);

Value* valueAddScalar(Value* a, double b);

Value* valueMul(Value* a, Value* b);
Value* valueMulScalar(Value* a, double b);

Value* valuePow(Value* a, int n);

Value* valueNeg(Value* a);

Value* valueSub(Value* a, Value* b);
Value* valueSubScalar(Value* a, double b);

Value* valueTruediv(Value* a, Value* b);

Value* valueTruedivScalar(Value* a, double b);
Value* valueScalarTruediv(double b, Value* a);
void backward(Value* root);

void free_value(Value* v);

#endif
