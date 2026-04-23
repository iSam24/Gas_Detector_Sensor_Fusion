%module MLX90640
%include "stdint.i"
%include "std_vector.i"

%{
int setup(int fps);
void cleanup(void);
float * get_frame(void);
std::vector<float> capture_frames(int num_frames);
%}

namespace std {
    %template(FloatVector) vector<float>;
}

%typemap(out) float *get_frame %{
    $result = PyList_New(768);
    for (int i = 0; i < 768; ++i) {
        PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
    }
%}

int setup(int fps);
void cleanup(void);
float * get_frame(void);
std::vector<float> capture_frames(int num_frames);
