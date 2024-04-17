#include <stdio.h>
#include<tensorflow/c/c_api.h>
#include <tensorflow/c/tf_buffer.h>

void free_buffer(void* data, size_t length) { free(data); }

void deallocator(void *ptr, size_t len, void* arg) { free((void*)ptr); }

TF_Buffer *read_file(const char *file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  void *data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;

  return buf;
}

int main(int argc, char **argv) {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());

  char filename[320];
  strcpy(filename, "./slopemodel.pb");
  

  TF_Buffer *graph_def = read_file(filename);
  TF_Graph *graph = TF_NewGraph();
  TF_Status *status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
    printf("ERROR: unable to import graph\n");
    return 1;
  }

  TF_SessionOptions *opt = TF_NewSessionOptions();
  TF_Session *session = TF_NewSession(graph, opt, status);
  TF_DeleteSessionOptions(opt);
  if (TF_GetCode(status) != TF_OK) {
    printf("ERROR: unable to import graph\n");
    return 1;
  }

  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to run restore_op: %s\n",
            TF_Message(status));
    return 1;
  }

  // two samples
	int SEQ_LENGTH = 21;
	float sampledata0[21] = { 1., 0.96713615, 0.88262911, 0.74178404, 0.69953052,
	0.66666667, 0.6713615, 0.57746479, 0.48826291, 0.50234742,
	0.5399061, 0.43661972, 0.33333333, 0.342723, 0.25821596,
	0.24882629, 0.19248826, 0.16901408, 0.07511737, 0.,
	0.03286385 };

	float sampledata1[21] = { 0.        , 0.0257732 , 0.01030928, 0.1185567 , 0.20618557,
			0.28865979, 0.30412371, 0.33505155, 0.36597938, 0.42268041,
			0.43814433, 0.55670103, 0.73195876, 0.69587629, 0.77835052,
			0.85051546, 0.8814433 , 0.81443299, 0.92783505, 0.98969072,
			1. };

	TF_Operation* input_op = TF_GraphOperationByName(graph, "dense_1_input");
  printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));

	// Gerenate input
	if (input_op == NULL) {
		printf("operattion not found\n");
		exit(0);
	}

  float *raw_input_data = (float *)malloc(SEQ_LENGTH * sizeof(float));
  for (int i = 0; i < 21; i++) raw_input_data[i] = sampledata1[i];

  int64_t *raw_input_dims = (int64_t*)malloc(2 * sizeof(int64_t));
  raw_input_dims[0] = 1;
  raw_input_dims[1] = 21;

  TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, raw_input_dims, 2, raw_input_data,
                                         SEQ_LENGTH * sizeof(float), &deallocator, NULL);

  TF_Output *run_inputs = (TF_Output *)malloc(1 * sizeof(TF_Output));
  run_inputs[0].oper = input_op;
  run_inputs[0].index = 0;

  TF_Tensor **run_input_tensors = (TF_Tensor **)malloc(1 * sizeof(TF_Tensor *));
  run_input_tensors[0] = input_tensor;

  // Prepare output
  TF_Operation *output_op = TF_GraphOperationByName(graph, "dense_2/Sigmoid");
  if (output_op == NULL) {
		printf("output operation not found\n");
		exit(0);
	}

  printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));

  TF_Output *run_outputs = (TF_Output *)malloc(1 * sizeof(TF_Output));
  run_outputs[0].oper = output_op;
  run_outputs[0].index = 0;

  TF_Tensor **run_output_tensors = (TF_Tensor **)malloc(1 * sizeof(TF_Output));
  float *raw_output_data = (float *)malloc(1 * sizeof(float));
  raw_output_data[0] = 1.f;
  int64_t *raw_output_dims = (int64_t *)malloc(1 * sizeof(int64_t));
  raw_output_dims[0] = 1.f;

  TF_Tensor *output_tensor = TF_NewTensor(TF_FLOAT, raw_output_dims, 1, raw_output_data, 
                                          1 * sizeof(float), &deallocator, NULL);
  run_output_tensors[0] = output_tensor;


  // Run network
  TF_SessionRun(session, NULL, run_inputs, run_input_tensors, 1, run_outputs, run_output_tensors,
                1, NULL, 0, NULL, status);

  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
    return 1;
  }

  void *output_data = TF_TensorData(run_output_tensors[0]);
  printf("output %f\n", ((float *)output_data)[0]);

  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);

  TF_DeleteStatus(status);
  TF_DeleteBuffer(graph_def);

  TF_DeleteGraph(graph);

  free((void*)raw_input_data);
  free((void*)raw_input_dims);
  free((void*)run_inputs);;

  return 0;
}
