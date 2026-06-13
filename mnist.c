#define CGRAD_IMPLEMENTATION
#include "cgrad.h"
#include <time.h>

// MNIST IDX file loader
static int read_int32_be(FILE* f)
{
  unsigned char buf[4];
  fread(buf, 1, 4, f);
  return ((int)buf[0] << 24) | ((int)buf[1] << 16) | ((int)buf[2] << 8) | (int)buf[3];
}

static unsigned char* read_idx_images(const char* path, int* out_n, int* out_rows, int* out_cols)
{
  FILE* f = fopen(path, "rb");
  if (!f) {
    printf("ERROR: cannot open %s\n", path);
    exit(1);
  }
  int magic = read_int32_be(f);
  if (magic != 0x0803) {
    printf("ERROR: bad magic 0x%08X in %s\n", magic, path);
    exit(1);
  }
  int n = read_int32_be(f);
  int rows = read_int32_be(f);
  int cols = read_int32_be(f);
  int size = n * rows * cols;
  unsigned char* data = malloc(size);
  fread(data, 1, size, f);
  fclose(f);
  *out_n = n;
  *out_rows = rows;
  *out_cols = cols;
  return data;
}

static unsigned char* read_idx_labels(const char* path, int* out_n)
{
  FILE* f = fopen(path, "rb");
  if (!f) {
    printf("ERROR: cannot open %s\n", path);
    exit(1);
  }
  int magic = read_int32_be(f);
  if (magic != 0x0801) {
    printf("ERROR: bad magic 0x%08X in %s\n", magic, path);
    exit(1);
  }
  int n = read_int32_be(f);
  unsigned char* data = malloc(n);
  fread(data, 1, n, f);
  fclose(f);
  *out_n = n;
  return data;
}

// 将像素值 [0,255] 归一化到 [0,1]
static float* normalize_pixels(const unsigned char* raw, int n, int rows, int cols)
{
  int size = n * rows * cols;
  float* out = malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
    out[i] = (float)raw[i] / 255.0f;
  return out;
}

// 创建 one-hot 标签
static float* make_onehot(const unsigned char* labels, int n, int num_classes)
{
  float* out = calloc(n * num_classes, sizeof(float));
  for (int i = 0; i < n; i++)
    out[i * num_classes + labels[i]] = 1.0f;
  return out;
}

// 获取预测类别
static int predict_class(Tensor* logits_2d)
{
  int best = 0;
  float best_val = logits_2d->data[0];
  for (int i = 1; i < 10; i++) {
    if (logits_2d->data[i] > best_val) {
      best_val = logits_2d->data[i];
      best = i;
    }
  }
  return best;
}

// 从 one-hot 标签获取真实类别
static int label_class(const float* onehot, int num_classes)
{
  for (int i = 0; i < num_classes; i++)
    if (onehot[i] == 1.0f)
      return i;
  return 0;
}

int main()
{
  srand((unsigned)time(NULL));

  // 超参数
  const int INPUT_SIZE = 784; // 28x28
  const int HIDDEN_SIZE = 128;
  const int NUM_CLASSES = 10;
  const int EPOCHS = 5;
  const float LEARNING_RATE = 0.01f;
  // 设为 0 表示用全部数据；设为较小值可以快速测试
    const int MAX_TRAIN = 0;

  // 加载数据
  printf("Loading MNIST data...\n");
  int train_n, test_n, rows, cols;
  unsigned char* train_images_raw = read_idx_images("data/train-images-idx3-ubyte", &train_n, &rows, &cols);
  unsigned char* train_labels_raw = read_idx_labels("data/train-labels-idx1-ubyte", &train_n);
  unsigned char* test_images_raw = read_idx_images("data/t10k-images-idx3-ubyte", &test_n, &rows, &cols);
  unsigned char* test_labels_raw = read_idx_labels("data/t10k-labels-idx1-ubyte", &test_n);
  printf("  Train: %d images, Test: %d images\n", train_n, test_n);

  float* train_images = normalize_pixels(train_images_raw, train_n, rows, cols);
  float* train_labels = make_onehot(train_labels_raw, train_n, NUM_CLASSES);
  float* test_images = normalize_pixels(test_images_raw, test_n, rows, cols);

  free(train_images_raw);
  free(test_images_raw);

  // 创建参数
  Tensor* W1 = cg_randn((int[]){INPUT_SIZE, HIDDEN_SIZE}, 2, 0.01f);
  Tensor* b1 = cg_zeros((int[]){1, HIDDEN_SIZE}, 2);
  Tensor* W2 = cg_randn((int[]){HIDDEN_SIZE, NUM_CLASSES}, 2, 0.01f);
  Tensor* b2 = cg_zeros((int[]){1, NUM_CLASSES}, 2);

  printf("  W1: (%d,%d), W2: (%d,%d)\n", INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES);

  // 训练
  int inp_shape[] = {1, INPUT_SIZE};
  int label_shape[] = {NUM_CLASSES};

  printf("\nStarting training (%d epochs, lr=%.4f)...\n\n", EPOCHS, LEARNING_RATE);

  time_t start_time = time(NULL);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float total_loss = 0.0f;
    int correct = 0;
    int n_samples = MAX_TRAIN > 0 ? MAX_TRAIN : train_n;

    for (int i = 0; i < n_samples; i++) {
      // 准备输入 (1, 784)
      float inp_data[INPUT_SIZE];
      memcpy(inp_data, train_images + i * INPUT_SIZE, INPUT_SIZE * sizeof(float));
      Tensor* input = cg_tensor(inp_data, inp_shape, 2);

      // 前向传播
      // z1 = input @ W1, shape (1, 128)
      Tensor* z1 = cg_matmul(input, W1);
      // h1 = z1 + b1, shape (1, 128)
      Tensor* h1 = cg_add(z1, b1);
      // a1 = relu(h1), shape (1, 128)
      Tensor* a1 = cg_relu(h1);
      // z2 = a1 @ W2, shape (1, 10)
      Tensor* z2 = cg_matmul(a1, W2);
      // logits = z2 + b2, shape (1, 10)
      Tensor* logits = cg_add(z2, b2);

      // 准备标签 (10,)
      float lbl_data[NUM_CLASSES];
      memcpy(lbl_data, train_labels + i * NUM_CLASSES, NUM_CLASSES * sizeof(float));
      Tensor* label = cg_tensor(lbl_data, label_shape, 1);

      // 计算损失
      Tensor* loss = cg_softmax_cross_entropy(logits, label);
      float loss_val = loss->data[0];
      total_loss += loss_val;

      // 反向传播
      cg_zero_grad(W1);
      cg_zero_grad(b1);
      cg_zero_grad(W2);
      cg_zero_grad(b2);
      cg_backward(loss);

      // SGD 更新
      cg_sgd_step(W1, LEARNING_RATE);
      cg_sgd_step(b1, LEARNING_RATE);
      cg_sgd_step(W2, LEARNING_RATE);
      cg_sgd_step(b2, LEARNING_RATE);

      // 统计准确率
      int pred = predict_class(logits);
      int true_class = label_class(lbl_data, NUM_CLASSES);
      if (pred == true_class)
        correct++;

      // 释放中间张量
      cg_free(loss);
      cg_free(label);
      cg_free(logits);
      cg_free(z2);
      cg_free(a1);
      cg_free(h1);
      cg_free(z1);
      cg_free(input);

      // 进度
      if ((i + 1) % 1000 == 0) {
        float avg_loss = total_loss / (i + 1);
        float acc = 100.0f * correct / (i + 1);
        printf("  Epoch %d | Step %d/%d | Loss: %.4f | Acc: %.2f%%\n",
               epoch + 1, i + 1, n_samples, avg_loss, acc);
        fflush(stdout);
      }
    }

    float avg_loss = total_loss / n_samples;
    float train_acc = 100.0f * correct / n_samples;

    // 测试集评估
    int test_correct = 0;
    for (int i = 0; i < test_n; i++) {
      float inp_data[INPUT_SIZE];
      memcpy(inp_data, test_images + i * INPUT_SIZE, INPUT_SIZE * sizeof(float));
      Tensor* input = cg_tensor(inp_data, inp_shape, 2);

      Tensor* z1 = cg_matmul(input, W1);
      Tensor* h1 = cg_add(z1, b1);
      Tensor* a1 = cg_relu(h1);
      Tensor* z2 = cg_matmul(a1, W2);
      Tensor* logits = cg_add(z2, b2);

      int pred = predict_class(logits);
      if (pred == test_labels_raw[i])
        test_correct++;

      cg_free(logits);
      cg_free(z2);
      cg_free(a1);
      cg_free(h1);
      cg_free(z1);
      cg_free(input);
    }
    float test_acc = 100.0f * test_correct / test_n;

    int elapsed = (int)(time(NULL) - start_time);
    printf("\n=== Epoch %d done | Train Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%% | Time: %ds ===\n\n",
           epoch + 1, avg_loss, train_acc, test_acc, elapsed);
    fflush(stdout);
  }

  int total_elapsed = (int)(time(NULL) - start_time);
  printf("Training completed in %ds\n", total_elapsed);

  // 清理
  cg_free(W1);
  cg_free(b1);
  cg_free(W2);
  cg_free(b2);
  free(train_images);
  free(train_labels);
  free(train_labels_raw);
  free(test_images);
  free(test_labels_raw);

  printf("Done.\n");
  return 0;
}
