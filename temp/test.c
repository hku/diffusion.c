#include <emscripten.h>
#include <stdlib.h>

// Function to create and retrieve the array pointer
float* getSample() {
    float (*array)[2][2][2] = malloc(sizeof(float[2][2][2]));
    if (!array) return NULL;  // Check if memory allocation succeeded

    // Initialize the array
    (*array)[0][0][0] = 1.1f; (*array)[0][0][1] = 2.2f;
    // ... initialize other elements ...

    return &(*array)[0][0][0];
}

// Function to retrieve the dimensions
void getDimensions(int* dimensions) {
    dimensions[0] = 2;
    dimensions[1] = 2;
    dimensions[2] = 2;
}

// Function to free the array once done
EMSCRIPTEN_KEEPALIVE
void freeArray(float* array) {
    free(array);
}
