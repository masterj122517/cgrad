CC = cc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

BUILD_DIR = build

.PHONY: all clean test mnist fashion

all: $(BUILD_DIR)/test $(BUILD_DIR)/mnist $(BUILD_DIR)/fashion_mnist

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test: test/test.c cgrad.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test/test.c $(LDFLAGS)

$(BUILD_DIR)/mnist: test/mnist.c cgrad.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test/mnist.c $(LDFLAGS)

$(BUILD_DIR)/fashion_mnist: test/fashion_mnist.c cgrad.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test/fashion_mnist.c $(LDFLAGS)

test: $(BUILD_DIR)/test
	$(BUILD_DIR)/test

mnist: $(BUILD_DIR)/mnist
	$(BUILD_DIR)/mnist

fashion: $(BUILD_DIR)/fashion_mnist
	$(BUILD_DIR)/fashion_mnist

clean:
	rm -rf $(BUILD_DIR)
