#include "engine.h"
#include <stdio.h>

int main(void)
{
  // a = Value(-4.0)
  Value* a = value_new(-4.0);

  // b = Value(2.0)
  Value* b = value_new(2.0);

  // c = a + b
  Value* c = valueAdd(a, b);

  // d = a * b + b**3
  Value* d = valueAdd(
      valueMul(a, b),
      valuePow(b, 3.0));

  // c += c + 1
  c = valueAdd(c, valueAddScalar(c, 1.0));

  // c += 1 + c + (-a)
  c = valueAdd(
      c,
      valueAdd(
          valueAddScalar(c, 1.0),
          valueMulScalar(a, -1.0)));

  // d += d * 2 + (b + a).relu()
  d = valueAdd(
      d,
      valueAdd(
          valueMulScalar(d, 2.0),
          relu(valueAdd(b, a))));

  // d += 3 * d + (b - a).relu()
  d = valueAdd(
      d,
      valueAdd(
          valueMulScalar(d, 3.0),
          relu(valueSub(b, a))));

  // e = c - d
  Value* e = valueSub(c, d);

  // f = e**2
  Value* f = valuePow(e, 2.0);

  // g = f / 2.0
  Value* g = valueTruedivScalar(f, 2.0);

  // g += 10.0 / f
  g = valueAdd(
      g,
      valueScalarTruediv(10.0, f));

  // forward result
  printf("%.4f\n", g->data); // 应该 ≈ 24.7041

  // backward
  backward(g);

  printf("%.4f\n", a->grad); // ≈ 138.8338
  printf("%.4f\n", b->grad); // ≈ 645.5773

  return 0;
}
