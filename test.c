#define CGRAD_IMPLEMENTATION
#include "cgrad.h"
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do {                          \
    tests_run++;                                     \
    printf("  %s ... ", #name);                      \
    fflush(stdout);                                  \
    test_##name();                                   \
    printf("PASSED\n");                              \
    tests_passed++;                                  \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, msg) do {                              \
    float va = (a), vb = (b);                                        \
    if (fabsf(va - vb) > 1e-4) {                                     \
        printf("\n  FAIL: %s: expected %.6f, got %.6f\n", msg, vb, va); \
        exit(1);                                                     \
    }                                                                \
} while(0)

TEST(tensor_create)
{
    int shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = cg_tensor(data, shape, 1);
    ASSERT_FLOAT_EQ(t->data[0], 1.0f, "data[0]");
    ASSERT_FLOAT_EQ(t->data[1], 2.0f, "data[1]");
    ASSERT_FLOAT_EQ(t->data[2], 3.0f, "data[2]");
    ASSERT_FLOAT_EQ(t->ndim, 1, "ndim");
    ASSERT_FLOAT_EQ(t->shape[0], 3, "shape[0]");
    cg_free(t);
}

TEST(add_forward)
{
    int shape[] = {4};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor* a = cg_tensor(a_data, shape, 1);
    Tensor* b = cg_tensor(b_data, shape, 1);
    Tensor* c = cg_add(a, b);
    ASSERT_FLOAT_EQ(c->data[0], 6.0f, "c[0]");
    ASSERT_FLOAT_EQ(c->data[1], 8.0f, "c[1]");
    ASSERT_FLOAT_EQ(c->data[2], 10.0f, "c[2]");
    ASSERT_FLOAT_EQ(c->data[3], 12.0f, "c[3]");
    cg_free(c);
    cg_free(b);
    cg_free(a);
}

TEST(add_backward)
{
    int shape[] = {1};
    float a_data[] = {2.0f};
    float b_data[] = {3.0f};
    Tensor* a = cg_tensor(a_data, shape, 1);
    Tensor* b = cg_tensor(b_data, shape, 1);
    Tensor* c = cg_add(a, b);
    Tensor* s = cg_sum(c);
    cg_backward(s);
    ASSERT_FLOAT_EQ(a->grad[0], 1.0f, "a.grad");
    ASSERT_FLOAT_EQ(b->grad[0], 1.0f, "b.grad");
    cg_free(s);
    cg_free(c);
    cg_free(b);
    cg_free(a);
}

TEST(matmul_forward)
{
    int shape_a[] = {2, 3};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* a = cg_tensor(a_data, shape_a, 2);

    int shape_b[] = {3, 2};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Tensor* b = cg_tensor(b_data, shape_b, 2);

    Tensor* c = cg_matmul(a, b);
    ASSERT_FLOAT_EQ(c->data[0], 58.0f, "c[0,0]");
    ASSERT_FLOAT_EQ(c->data[1], 64.0f, "c[0,1]");
    ASSERT_FLOAT_EQ(c->data[2], 139.0f, "c[1,0]");
    ASSERT_FLOAT_EQ(c->data[3], 154.0f, "c[1,1]");

    cg_free(c);
    cg_free(b);
    cg_free(a);
}

TEST(matmul_backward)
{
    int shape_a[] = {2, 3};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* a = cg_tensor(a_data, shape_a, 2);

    int shape_b[] = {3, 2};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Tensor* b = cg_tensor(b_data, shape_b, 2);

    Tensor* c = cg_matmul(a, b);
    Tensor* s = cg_sum(c);
    cg_backward(s);

    ASSERT_FLOAT_EQ(a->grad[0], 15.0f, "a.grad[0]");
    ASSERT_FLOAT_EQ(a->grad[1], 19.0f, "a.grad[1]");
    ASSERT_FLOAT_EQ(a->grad[2], 23.0f, "a.grad[2]");
    ASSERT_FLOAT_EQ(a->grad[3], 15.0f, "a.grad[3]");
    ASSERT_FLOAT_EQ(a->grad[4], 19.0f, "a.grad[4]");
    ASSERT_FLOAT_EQ(a->grad[5], 23.0f, "a.grad[5]");

    ASSERT_FLOAT_EQ(b->grad[0], 5.0f, "b.grad[0]");
    ASSERT_FLOAT_EQ(b->grad[1], 5.0f, "b.grad[1]");
    ASSERT_FLOAT_EQ(b->grad[2], 7.0f, "b.grad[2]");
    ASSERT_FLOAT_EQ(b->grad[3], 7.0f, "b.grad[3]");
    ASSERT_FLOAT_EQ(b->grad[4], 9.0f, "b.grad[4]");
    ASSERT_FLOAT_EQ(b->grad[5], 9.0f, "b.grad[5]");

    cg_free(s);
    cg_free(c);
    cg_free(b);
    cg_free(a);
}

TEST(relu_forward)
{
    int shape[] = {4};
    float data[] = {-1.0f, 0.0f, 2.0f, -3.0f};
    Tensor* a = cg_tensor(data, shape, 1);
    Tensor* r = cg_relu(a);
    ASSERT_FLOAT_EQ(r->data[0], 0.0f, "relu(-1)");
    ASSERT_FLOAT_EQ(r->data[1], 0.0f, "relu(0)");
    ASSERT_FLOAT_EQ(r->data[2], 2.0f, "relu(2)");
    ASSERT_FLOAT_EQ(r->data[3], 0.0f, "relu(-3)");
    cg_free(r);
    cg_free(a);
}

TEST(relu_backward)
{
    int shape[] = {3};
    float data[] = {-1.0f, 2.0f, -3.0f};
    Tensor* a = cg_tensor(data, shape, 1);
    Tensor* r = cg_relu(a);
    Tensor* s = cg_sum(r);
    cg_backward(s);
    ASSERT_FLOAT_EQ(a->grad[0], 0.0f, "grad at x<0");
    ASSERT_FLOAT_EQ(a->grad[1], 1.0f, "grad at x>0");
    ASSERT_FLOAT_EQ(a->grad[2], 0.0f, "grad at x<0 again");
    cg_free(s);
    cg_free(r);
    cg_free(a);
}

TEST(sum_forward)
{
    int shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* a = cg_tensor(data, shape, 1);
    Tensor* s = cg_sum(a);
    ASSERT_FLOAT_EQ(s->data[0], 10.0f, "sum");
    cg_free(s);
    cg_free(a);
}

TEST(softmax_cross_entropy_forward)
{
    int shape[] = {3};
    float logits_data[] = {2.0f, 1.0f, 0.0f};
    float label_data[] = {0.0f, 1.0f, 0.0f};
    Tensor* logits = cg_tensor(logits_data, shape, 1);
    Tensor* labels = cg_tensor(label_data, shape, 1);
    Tensor* loss = cg_softmax_cross_entropy(logits, labels);

    float predicted_loss = loss->data[0];
    float expected = -logf(2.718281828f / (7.389056099f + 2.718281828f + 1.0f) + 1e-7f);
    ASSERT_FLOAT_EQ(predicted_loss, expected, "loss value");

    cg_free(loss);
    cg_free(labels);
    cg_free(logits);
}

TEST(softmax_cross_entropy_backward)
{
    int shape[] = {3};
    float logits_data[] = {2.0f, 1.0f, 0.0f};
    float label_data[] = {0.0f, 1.0f, 0.0f};
    Tensor* logits = cg_tensor(logits_data, shape, 1);
    Tensor* labels = cg_tensor(label_data, shape, 1);
    Tensor* loss = cg_softmax_cross_entropy(logits, labels);
    cg_backward(loss);

    ASSERT_FLOAT_EQ(logits->grad[0], 0.66524094f, "grad[0]");
    ASSERT_FLOAT_EQ(logits->grad[1], -0.75527155f, "grad[1]");
    ASSERT_FLOAT_EQ(logits->grad[2], 0.09003057f, "grad[2]");

    cg_free(loss);
    cg_free(labels);
    cg_free(logits);
}

TEST(sgd_step)
{
    int shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = cg_tensor(data, shape, 1);
    t->grad[0] = 0.1f;
    t->grad[1] = 0.2f;
    t->grad[2] = 0.3f;
    cg_sgd_step(t, 0.5f);
    ASSERT_FLOAT_EQ(t->data[0], 1.0f - 0.5f * 0.1f, "data[0]");
    ASSERT_FLOAT_EQ(t->data[1], 2.0f - 0.5f * 0.2f, "data[1]");
    ASSERT_FLOAT_EQ(t->data[2], 3.0f - 0.5f * 0.3f, "data[2]");
    cg_free(t);
}

TEST(zero_grad)
{
    int shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = cg_tensor(data, shape, 1);
    t->grad[0] = 5.0f;
    t->grad[1] = 10.0f;
    t->grad[2] = 15.0f;
    cg_zero_grad(t);
    ASSERT_FLOAT_EQ(t->grad[0], 0.0f, "grad[0] zero");
    ASSERT_FLOAT_EQ(t->grad[1], 0.0f, "grad[1] zero");
    ASSERT_FLOAT_EQ(t->grad[2], 0.0f, "grad[2] zero");
    cg_free(t);
}

TEST(randn_basic)
{
    int shape[] = {100};
    Tensor* t = cg_randn(shape, 1, 1.0f);
    ASSERT_FLOAT_EQ(t->ndim, 1, "ndim");
    ASSERT_FLOAT_EQ(t->shape[0], 100, "shape");
    int non_zero = 0;
    for (int i = 0; i < 100; i++)
        if (fabsf(t->data[i]) > 1e-6f) non_zero++;
    ASSERT_FLOAT_EQ(non_zero > 50, 1, "non_zero count");
    cg_free(t);
}

TEST(gradient_accumulation)
{
    int shape[] = {1};
    float x_data[] = {3.0f};
    Tensor* x = cg_tensor(x_data, shape, 1);
    Tensor* a = cg_add(x, x);
    Tensor* b = cg_add(x, x);
    Tensor* c = cg_add(a, b);
    Tensor* s = cg_sum(c);
    cg_backward(s);
    ASSERT_FLOAT_EQ(x->grad[0], 4.0f, "x.grad from multi-path");

    cg_free(s);
    cg_free(c);
    cg_free(b);
    cg_free(a);
    cg_free(x);
}

TEST(simple_mlp_forward)
{
    int in_shape[] = {3};
    float in_data[] = {1.0f, 2.0f, 3.0f};
    Tensor* input = cg_tensor(in_data, in_shape, 1);

    int w1_shape[] = {3, 4};
    float w1_data[] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f};
    Tensor* w1 = cg_tensor(w1_data, w1_shape, 2);

    int b1_shape[] = {4};
    float b1_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
    Tensor* b1 = cg_tensor(b1_data, b1_shape, 1);

    int inp2d_shape[] = {1, 3};
    float inp2d_data[] = {1.0f, 2.0f, 3.0f};
    Tensor* inp2d = cg_tensor(inp2d_data, inp2d_shape, 2);

    Tensor* hidden_raw = cg_matmul(inp2d, w1);

    int b1_2d_shape[] = {1, 4};
    float b1_2d_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
    Tensor* b1_2d = cg_tensor(b1_2d_data, b1_2d_shape, 2);

    Tensor* hidden = cg_add(hidden_raw, b1_2d);

    ASSERT_FLOAT_EQ(hidden->data[0], 3.9f, "hidden[0]");
    ASSERT_FLOAT_EQ(hidden->data[1], 4.6f, "hidden[1]");
    ASSERT_FLOAT_EQ(hidden->data[2], 5.3f, "hidden[2]");
    ASSERT_FLOAT_EQ(hidden->data[3], 6.0f, "hidden[3]");

    Tensor* act = cg_relu(hidden);
    ASSERT_FLOAT_EQ(act->data[0], 3.9f, "act[0]");
    ASSERT_FLOAT_EQ(act->data[1], 4.6f, "act[1]");
    ASSERT_FLOAT_EQ(act->data[2], 5.3f, "act[2]");
    ASSERT_FLOAT_EQ(act->data[3], 6.0f, "act[3]");

    int w2_shape[] = {4, 2};
    float w2_data[] = {0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f};
    Tensor* w2 = cg_tensor(w2_data, w2_shape, 2);

    Tensor* logits = cg_matmul(act, w2);
    ASSERT_FLOAT_EQ(logits->data[0], 16.54f, "logits[0]");
    ASSERT_FLOAT_EQ(logits->data[1], 18.52f, "logits[1]");

    int label_shape[] = {2};
    float label_data[] = {1.0f, 0.0f};
    Tensor* label = cg_tensor(label_data, label_shape, 1);

    Tensor* loss = cg_softmax_cross_entropy(logits, label);
    cg_backward(loss);

    ASSERT_FLOAT_EQ(w2->grad[0] != 0.0f, 1, "w2 grad exists");
    ASSERT_FLOAT_EQ(w1->grad[0] != 0.0f, 1, "w1 grad exists");

    cg_free(loss);
    cg_free(label);
    cg_free(logits);
    cg_free(w2);
    cg_free(act);
    cg_free(hidden);
    cg_free(b1_2d);
    cg_free(hidden_raw);
    cg_free(inp2d);
    cg_free(b1);
    cg_free(w1);
    cg_free(input);
}

int main()
{
    srand(42);
    printf("Running cgrad tests...\n\n");

    RUN_TEST(tensor_create);
    RUN_TEST(add_forward);
    RUN_TEST(add_backward);
    RUN_TEST(matmul_forward);
    RUN_TEST(matmul_backward);
    RUN_TEST(relu_forward);
    RUN_TEST(relu_backward);
    RUN_TEST(sum_forward);
    RUN_TEST(softmax_cross_entropy_forward);
    RUN_TEST(softmax_cross_entropy_backward);
    RUN_TEST(sgd_step);
    RUN_TEST(zero_grad);
    RUN_TEST(randn_basic);
    RUN_TEST(gradient_accumulation);
    RUN_TEST(simple_mlp_forward);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
