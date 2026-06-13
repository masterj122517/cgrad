#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#include "nob.h"

#define BUILD_FOLDER "build/"
#define TEST_FOLDER "test/"

int main(int argc, char** argv)
{
  NOB_GO_REBUILD_URSELF(argc, argv);

  Cmd cmd = {0};
  Procs procs = {0};

  if (!mkdir_if_not_exists(BUILD_FOLDER))
    return 1;

  static struct
  {
    const char* bin_path;
    const char* src_path;
  } targets[] = {
      {.bin_path = BUILD_FOLDER "test", .src_path = TEST_FOLDER "test.c"},
      {.bin_path = BUILD_FOLDER "mnist", .src_path = TEST_FOLDER "mnist.c"},
      {.bin_path = BUILD_FOLDER "fashion_mnist", .src_path = TEST_FOLDER "fashion_mnist.c"},
  };

  // Spawn one async process per target collecting them to procs dynamic array
  for (size_t i = 0; i < ARRAY_LEN(targets); ++i) {
    nob_cc(&cmd);
    nob_cc_flags(&cmd);
    nob_cmd_append(&cmd, "-lm", "-O2"); // your extra flags
    nob_cc_output(&cmd, targets[i].bin_path);
    nob_cc_inputs(&cmd, targets[i].src_path);
    if (!cmd_run(&cmd, .async = &procs))
      return 1;
  }

  // Wait on all the async processes to finish and reset procs dynamic array to 0
  if (!procs_flush(&procs))
    return 1;

  return 0;
}
