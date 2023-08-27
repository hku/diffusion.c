#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>



//file operation such as O_RDONLY
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif


//some consant used in program
float sqrt_2_over_pi = sqrt(2.0 / M_PI);
int POOL_SIZE =2;
int SIZE_K = 3; //kernel size in conv layer
int SIZE_P = 1; //padding size in conv layer

typedef struct {
    int width; //image width
    int height; //image height
    int n_in; //input channel size: rgb
    int n_feature;
    int n_cfeature;
} Config;

typedef struct {
    int C; //channels
    int H; //height
    int W; //width
    int size; //size=C*H*W
} Shape3; //shape for 3d matrix

typedef struct {
    int size;
} Shape1;//shape for a vector (or flatten matrix)

typedef struct {
    float* weight;
    float* bias;
} LayerParams;

typedef struct {
    float* weight;
    float* bias;
    float* weight2;
    float* bias2;
    float* weight3;
    float* bias3;
} ResidualConvParams;

typedef struct {
    float* weight;
    float* bias;
    float* weight2;
    float* bias2;
} OutParams;

typedef struct {
    LayerParams* params_block1;
    LayerParams* params_block2;
} EmbFCParams;

typedef struct {
    ResidualConvParams* params_block1;
    ResidualConvParams* params_block2;
} UnetDownParams;

typedef struct {
    LayerParams* params_block0;
    ResidualConvParams* params_block1;
    ResidualConvParams* params_block2;
} UnetUpParams;

typedef struct {
    ResidualConvParams* params_block0; //init_conv
    UnetDownParams* params_block1; //down1
    UnetDownParams* params_block2; //down2
    EmbFCParams* timeembed1;
    EmbFCParams* timeembed2;
    EmbFCParams* contextembed1;
    EmbFCParams* contextembed2;
    LayerParams* params_up0;
    UnetUpParams* params_up1;
    UnetUpParams* params_up2;
    OutParams* params_out;
} ContextUnetParams;

int init_fc_weight(LayerParams* w, int n_in, int n_out,  float* ptr) {
   w->weight = ptr;
   int weight_size = n_in * n_out;
   ptr += weight_size; 
   w->bias = ptr;
   return weight_size + n_out;    
}

int init_embfc_weight(EmbFCParams* w, int n_in, int n_out,  float* ptr) {
    w->params_block1 = malloc(sizeof(LayerParams));
    w->params_block2 = malloc(sizeof(LayerParams));
    int block1_size = init_fc_weight(w->params_block1, n_in, n_out, ptr);
    ptr += block1_size;
    int block2_size = init_fc_weight(w->params_block2, n_out, n_out, ptr);
    return block1_size + block2_size;
}

int init_out_weight(OutParams* w, int n_in, int n_out, int size_k, float* ptr) {

    w->weight = ptr;

    int weight_size = 2*n_out * n_out * size_k * size_k;
    ptr += weight_size;
    w->bias = ptr;
    ptr += n_out;
    w->weight2 = ptr;
    int weight2_size = n_out * n_in * size_k * size_k;
    ptr += weight2_size;
    w->bias2 = ptr;

    return weight_size + n_out + weight2_size  + n_in;

}

int init_convtrans_weight(LayerParams* w, int n_in, int n_feature, int size_k,  float* ptr) {
   w->weight = ptr;
   int weight_size = n_in * n_feature * size_k * size_k;
   ptr += weight_size; 
   w->bias = ptr;
   return weight_size + n_feature;
}

int init_residual_weights(ResidualConvParams* w, int n_in, int n_feature, int size_k, bool is_res, float* f) {
    float* ptr = f;
    w->weight = ptr;

    int weight_size = n_in * n_feature * size_k * size_k;
    ptr += weight_size;
    w->bias = ptr;
    ptr += n_feature;
    w->weight2 = ptr;

    int weight2_size = n_feature * n_feature * size_k * size_k;

    ptr += weight2_size;
    
    w->bias2 = ptr;
    if(is_res && (n_in != n_feature)) {
        ptr += n_feature;
        w->weight3 = ptr;
        int weight3_size = n_in * n_feature * 1 * 1;
        ptr += weight3_size;
        w->bias3 = ptr;

        return weight_size + n_feature + weight2_size  + n_feature + weight3_size + n_feature;
    }

    return weight_size + n_feature + weight2_size  + n_feature;
}

int init_unetdown_weights(UnetDownParams* w, int n_in, int n_feature, int size_k, float* ptr){
    w->params_block1 = malloc(sizeof(ResidualConvParams));
    w->params_block2 = malloc(sizeof(ResidualConvParams));

    int block1_size = init_residual_weights(w->params_block1, n_in, n_feature, size_k, false, ptr);
    ptr += block1_size;
    int block2_size = init_residual_weights(w->params_block2, n_feature, n_feature, size_k, false, ptr);
    return block1_size + block2_size;
}

int init_unetup_weights(UnetUpParams* w, int n_in, int n_feature, int size_k, float* ptr){
    w->params_block0 = malloc(sizeof(LayerParams));
    w->params_block1 = malloc(sizeof(ResidualConvParams));
    w->params_block2 = malloc(sizeof(ResidualConvParams));

    int conv_trans_size_k = 2;

    int block0_size = init_convtrans_weight(w->params_block0, n_in, n_feature, conv_trans_size_k, ptr);
    ptr += block0_size;
    int block1_size = init_residual_weights(w->params_block1, n_feature, n_feature, size_k, false, ptr);
    ptr += block1_size;
    int block2_size = init_residual_weights(w->params_block2, n_feature, n_feature, size_k, false, ptr);
    return block0_size + block1_size + block2_size;
}

int init_context_unet_weights(ContextUnetParams* w, int n_in, int n_feature, int n_cfeature, int size_k, int up0_size_k, int up0_size_s, float* ptr){
    w->params_block0 = malloc(sizeof(ResidualConvParams));
    w->params_block1 = malloc(sizeof(UnetDownParams));
    w->params_block2 = malloc(sizeof(UnetDownParams));
    w->timeembed1 =malloc(sizeof(EmbFCParams));
    w->timeembed2 =malloc(sizeof(EmbFCParams));
    w->contextembed1 =malloc(sizeof(EmbFCParams));
    w->contextembed2 =malloc(sizeof(EmbFCParams));
    w->params_up0 =malloc(sizeof(LayerParams));
    w->params_up1 =malloc(sizeof(UnetUpParams));
    w->params_up2 =malloc(sizeof(UnetUpParams));
    w->params_out =malloc(sizeof(OutParams));

    int block0_size = init_residual_weights(w->params_block0, n_in, n_feature, size_k, true, ptr);
    ptr += block0_size;
    int block1_size = init_unetdown_weights(w->params_block1, n_feature, n_feature, size_k, ptr);
    ptr += block1_size;
    int block2_size = init_unetdown_weights(w->params_block2, n_feature, 2*n_feature, size_k, ptr);
    ptr += block2_size;
    int time1_size = init_embfc_weight(w->timeembed1, 1, 2*n_feature,  ptr);
    ptr += time1_size;
    int time2_size = init_embfc_weight(w->timeembed2, 1, n_feature,  ptr);
    ptr += time2_size;
    int context1_size = init_embfc_weight(w->contextembed1, n_cfeature, 2*n_feature,  ptr);
    ptr += context1_size;
    int context2_size = init_embfc_weight(w->contextembed2, n_cfeature, n_feature,  ptr);
    ptr += context2_size;
    int up0_size = init_convtrans_weight(w->params_up0, 2*n_feature, 2*n_feature, up0_size_k, ptr);
    ptr += up0_size;
    int up1_size = init_unetup_weights(w->params_up1, 4*n_feature, n_feature, size_k, ptr);
    ptr += up1_size;
    int up2_size = init_unetup_weights(w->params_up2, 2*n_feature, n_feature, size_k, ptr);
    ptr += up2_size;
    int out_size = init_out_weight(w->params_out, n_in, n_feature, size_k, ptr);

// (EmbFCParams* w, int n_in, int n_out,  float* ptr) 
    return block0_size + block1_size + block2_size + time1_size + time2_size + context1_size + context2_size + up0_size + up1_size + up2_size + out_size;
    
}

//get the value from a flatten array by the 3d coordinates (ic, i, j)
float get_value3(float* input, int ic, int i, int j, int H, int W, int p) {
    int x = i - p; //padding p
    int y = j - p; //padding p
    if (x < 0 || x >= H || y < 0 || y >= W) {
        return 0.0f;  // Padding value
    }
    return input[ic * H * W + x * W + y];
}

//get the value from a flatten array by the 4d coordinates (oc, ic, i, j)
float get_value4(float* input, int oc, int ic, int i, int j, int D, int H, int W, int p) {
    int x = i - p; //padding p
    int y = j - p; //padding p
    if (x < 0 || x >= H || y < 0 || y >= W) {
        return 0.0f;  // Padding value
    }
    return input[oc * D * H * W + ic * H * W + x * W + y];
}

float compute_mean(float* arr, int length) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += arr[i];
    }
    return sum / length;
}

// Helper function to compute the standard deviation of an array
float compute_std_dev(float* arr, int length, float mean) {
    float eps = 1e-05;

    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += (arr[i] - mean) * (arr[i] - mean);
    }
    return sqrt(sum/length  + eps);
}

float* add(float*a, float* b, int size){
    float* c = malloc(size*sizeof(float));
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

float* concat_channel(float* x, float* y, int Cx, int Cy, int H, int W) {

    // Calculate the total length of the flattened arrays
    int len_x = Cx * H * W;
    int len_y = Cy * H * W;
    int len_z = len_x + len_y;

    // Allocate memory for the concatenated array
    float* z = (float*)malloc(len_z * sizeof(float));

    // Flatten and copy elements from xx to z
    for (int i = 0; i < Cx; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                z[i * H * W + j * W + k] = x[i*H*W + j*W + k];
            }
        }
    }

    // Flatten and copy elements from yy to z
    for (int i = 0; i < Cy; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                z[len_x + i * H * W + j * W + k] = y[i*H*W + j*W + k];
            }
        }
    }

    return z;
}

void gelu(float* xx, int L) {

    float a = 0.044715;
    for (int i =0; i<L; i++) {
        float x = xx[i];
        float inner_tanh = sqrt_2_over_pi * (x + a * x*x*x);
        xx[i] = 0.5 * x * (1.0 + tanh(inner_tanh));
    }
}

void relu(float *xx, int L) {
    for (int i =0; i<L; i++) {
        if(xx[i]<0) {xx[i] = 0;}
    }
}

void elm_linear(float* x, float* w, float* b, int C, int HW) {
    for(int c=0; c<C; c++) {
        for(int i=0; i<HW; i++) {
            x[c*HW + i] = x[c*HW + i] * w[c] + b[c];
        }
    }
}

Shape1 linear(float* weight, float* bias, float* x, float** y, int n_out, int n_in) {

    *y = (float*)malloc(n_out * sizeof(float));

    float sum;
    #pragma omp parallel private(sum)
    for (int i = 0; i < n_out; i++) {
        sum = 0.0;
        for (int j = 0; j < n_in; j++) {
            sum += x[j] * weight[i * n_in + j];
        }
        (*y)[i] = sum + bias[i];
    }

    Shape1 shape_y;
    shape_y.size = n_out;
    return shape_y;
}

Shape1 EmbFC(EmbFCParams* w, float* x, float** y, int n_in, int n_out) {

    LayerParams* params_block1 = w->params_block1;
    LayerParams* params_block2 = w->params_block2;

    float* weight = params_block1->weight;
    float* bias = params_block1->bias;

    float* weight2 = params_block2->weight;
    float* bias2 = params_block2->bias;

    float* y1;
    Shape1 shape_y1 = linear(weight, bias, x, &y1, n_out, n_in);

    gelu(y1, shape_y1.size);

    Shape1 shape_y = linear(weight2, bias2, y1, &(*y), n_out, n_out);


    return shape_y;
}

Shape3 conv2d(float* weight, float* bias, float* x, float** y, int H, int W, int n_in, int n_out, int size_k, int size_p) {
    int height_y =  H + 2*size_p - (size_k-1);
    int width_y =  W + 2*size_p - (size_k-1);
    int size_y = n_out*height_y*width_y;

    *y = calloc(size_y, sizeof(float));
    int oc;
    float sum;

//    #pragma omp parallel for collapse(3) private(sum)
    for (oc = 0; oc < n_out; oc++) {
        for (int i = 0; i < height_y; i++) {
            for (int j = 0; j < width_y; j++) {
                sum = 0.0f;
                for (int ic = 0; ic < n_in; ic++) {
                    for (int ki = 0; ki < size_k; ki++) {
                        for (int kj = 0; kj < size_k; kj++) {
                            sum += get_value3(
                                x, ic, i+ki, j+kj, 
                                H, W, size_p
                            ) * get_value4(
                                weight, oc, ic, ki, kj, 
                                n_in, size_k, size_k, 0
                            );                            
                        }
                    }
                }
                sum += bias[oc];
                (*y)[oc*height_y*width_y+i*width_y+j] = sum;
            }
        }
    }

    Shape3 shape;
    shape.C = n_out;
    shape.H = height_y;
    shape.W = width_y;
    shape.size = size_y;
    return shape;

}

Shape3 convTrans2d(float* weight, float* bias, float* x, float** y, int H, int W, int n_in, int n_out, int size_k, int size_s) {
    // size_s is stride size
    // y must be zero intialized first! i.e. 
    // float* y = calloc(size_y, sizeof(float));;
    
    int height_y =  (H - 1)*size_s + size_k;
    int width_y =  (W - 1)*size_s + size_k;    
    int size_y = n_out*height_y*width_y;

    *y = calloc(size_y, sizeof(float));

    // Perform the transposed convolution operation
    for (int oc = 0; oc < n_out; oc++) {
        for (int ic = 0; ic < n_in; ic++) {
          for (int i = 0; i < H; i++) {
              for (int j = 0; j < W; j++) {
                      for (int m = 0; m < size_k; m++) {
                          for (int n = 0; n < size_k; n++) {
                              int h = i * size_s + m;
                              int w = j * size_s + n;
                              (*y)[oc*height_y*width_y + h*width_y + w] += get_value3(
                                  x, ic, i, j,
                                  H, W, 0
                              ) * get_value4(
                                  weight, ic, oc, m, n,
                                  n_out, size_k, size_k, 0
                              );
                              //nn.Conv2d kernel shape is (oc,ic,kh,kw), while
                              //nn.ConvTrans2d kernel shape is (ic, oc, kh, kw)
                              //the order affects how we read the weight and do the calculation.
                              // c1 = nn.Conv2d(8, 16, [4, 2])
                              // c2 = nn.ConvTranspose2d(8, 16, [4, 2])
                              // c1.shape = [16, 8, 4, 2]
                              // c2.shape = [8, 16, 4, 2]
                          }
                      }
              }
          }
        }
    }




    for (int oc = 0; oc < n_out; oc++) {
        for (int i = 0; i < height_y; i++) {
            for (int j = 0; j < width_y; j++) {
                (*y)[oc*height_y*width_y + i*width_y + j] += bias[oc];
            }
        }        
    }

    Shape3 shape;
    shape.C = n_out;
    shape.H = height_y;
    shape.W = width_y;
    shape.size = size_y;
    return shape;    

}

void batchNorm2d(float* x, int n_feature, int H, int W) {
    float epsilon = 1e-05;
    float mean;
    float var;
    for (int c = 0; c < n_feature; c++) {
        mean = 0.0;
        var = 0.0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                mean += get_value3(x, c, h, w, H, W, 0);
            }
        }
        mean /= (H*W);


        for (int h = 0; h <H; h++) {
            for (int w = 0; w < W; w++) {
                float d = get_value3(x, c, h, w, H, W, 0) - mean;
                var += d*d;
            }
        }
        var /= (H * W);

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                x[c*H*W + h*W + w] = (x[c*H*W + h*W + w] - mean) / sqrt(var + epsilon);
            }
        }        

    }
}

void groupNorm2d(float* input, int num_channels, int num_groups, int H, int W) {
    int channels_per_group = num_channels / num_groups;
    int size_each_group =  channels_per_group * H * W; 

    for (int i = 0; i < num_groups; i++) {
        float mean = compute_mean(input + i * size_each_group, size_each_group);
        float std_dev = compute_std_dev(input + i * size_each_group, size_each_group, mean);
        
        for (int j = 0; j < channels_per_group; j++) {
            for (int k = 0; k < H * W; k++) {
                int index = (i * channels_per_group + j) * H * W + k;
                input[index] = (input[index] - mean) / std_dev;
            }
        }
    }
}

Shape3 ResidualConvBlock(ResidualConvParams* w, float* x, float** y, int H, int W, int n_in, int n_out, int size_k, int size_p, bool is_res) {

    float* weight = w->weight;
    float* bias = w->bias;
    float* weight2 = w->weight2;
    float* bias2 = w->bias2;

    float* y2;

    Shape3 shape_y2=conv2d(weight, bias, x, &y2, H, W, n_in,n_out, size_k, size_p);
    batchNorm2d(y2, shape_y2.C, shape_y2.H, shape_y2.W);
    gelu(y2, shape_y2.size);

    Shape3 shape_y = conv2d(weight2, bias2, y2, &(*y), shape_y2.H, shape_y2.W, shape_y2.C, n_out, size_k, size_p);
    batchNorm2d(*y, shape_y.C, shape_y.H, shape_y.W);
    gelu(*y, shape_y.size);

    if(is_res) {
        if(n_in == n_out) {
            *y = add(*y, x, H*W*n_out);
        } else {
            float* weight3 = w->weight3;
            float* bias3 = w->bias3;
            float* shortcut;
            Shape3 shape_shortcut = conv2d(weight3, bias3, x, &shortcut, H, W, n_in, n_out, 1, 0);
            *y = add(*y, shortcut, shape_shortcut.size);
            free(shortcut);
        }
        for(int i = 0; i<shape_y.size; i++) {
            (*y)[i] = (*y)[i]/1.414;
        }
    }

    free(y2);

    return shape_y;
}

Shape3 Out(OutParams* w, float* x, float* x0, float** y, int H, int W, int n_in, int n_out, int size_k, int size_p) {
    
    float* x1 = concat_channel(x, x0, n_out, n_out, H, W);
    
    float* weight = w->weight;
    float* bias = w->bias;
    float* weight2 = w->weight2;
    float* bias2 = w->bias2;

    float* y1;
    Shape3 shape_y1 = conv2d(weight, bias, x1, &y1, H, W, 2*n_out, n_out, size_k, size_p);

    groupNorm2d(y1, n_out, 8, shape_y1.H, shape_y1.W);
    relu(y1, shape_y1.size);

    Shape3 shape_y = conv2d(weight2, bias2, y1, &(*y), shape_y1.H, shape_y1.W, shape_y1.C, n_in, size_k, size_p);

    free(y1);

    return shape_y;

}

Shape3 maxPool2d(float* x, float** y, int H, int W, int n_in, int size_k) {
    float max;

    int H_y = H/size_k;
    int W_y = W/size_k;
    int size_y = n_in*H_y * W_y;

    *y = (float*)malloc(size_y * sizeof(float));


//    #pragma omp parallel for collapse(3)
    for(int ic = 0; ic < n_in; ic++) {
        for(int i = 0; i < H; i += size_k) {
            for(int j = 0; j < W; j += size_k) {
                max = x[ic*H*W + i*W + j];
                for(int k = 0; k < size_k; k++) {
                    for(int l = 0; l < size_k; l++) {
                        float v = x[ic*H*W +(i + k)*W + (j + l)];
                        if(v > max) { max = v; }
                    }
                }
                (*y)[ic*H_y*W_y + (i/size_k)*W_y + j/size_k] = max;
            }
        }
    }

    Shape3 shape;
    shape.C = n_in;
    shape.H = H_y;
    shape.W = W_y;
    shape.size = size_y;
    return shape;
}

Shape3 avgPool2d(float* x, float** y, int H, int W, int n_in, int size_k) {

    int H_y = H/size_k;
    int W_y = W/size_k;
    int size_y = n_in*H_y * W_y;

    *y = (float*)malloc(size_y * sizeof(float));


//    #pragma omp parallel for collapse(3)
    for(int ic = 0; ic < n_in; ic++) {
        for(int i = 0; i < H; i += size_k) {
            for(int j = 0; j < W; j += size_k) {
                float sum = 0;
                for(int k = 0; k < size_k; k++) {
                    for(int l = 0; l < size_k; l++) {
                        sum += x[ic*H*W +(i + k)*W + (j + l)];
                    }
                }
                (*y)[ic*H_y*W_y + (i/size_k)*W_y + j/size_k] = sum/(size_k*size_k);
            }
        }
    }

    Shape3 shape;
    shape.C = n_in;
    shape.H = H_y;
    shape.W = W_y;
    shape.size = size_y;
    return shape;
}

Shape3 to_vec(float* x, float** y, int H, int W, int n_in, int size_k) {
    Shape3 yshape = avgPool2d(x, &(*y), H, W, n_in, size_k);
    gelu(*y, yshape.size);
    return yshape;
}

Shape3 Up0(LayerParams* w, float* x, float** y, int H, int W, int n_in, int n_out, int up0_size_k, int up0_size_s) {
    float* weight = w->weight;
    float* bias = w->bias;

    Shape3 yshape = convTrans2d(weight, bias, x, &(*y), H, W, n_in, n_out, up0_size_k, up0_size_s);

    groupNorm2d(*y, yshape.C, 8, yshape.H, yshape.W);

    relu(*y, yshape.size);

    return yshape;

}

Shape3 UnetDown(UnetDownParams* w, float* x, float** y, int H, int W, int n_in, int n_out, int size_k, int size_p) {
    
    ResidualConvParams* params_block1 = w->params_block1;
    ResidualConvParams* params_block2 = w->params_block2;

    float* y1;
    float* y2;

    Shape3 yshape1 = ResidualConvBlock(params_block1, x, &y1, H, W, n_in, n_out, size_k, size_p, false);
    Shape3 yshape2 = ResidualConvBlock(params_block2, y1, &y2, yshape1.H, yshape1.W, yshape1.C, n_out, size_k, size_p, false);
        
    Shape3 yshape = maxPool2d(y2, &(*y), yshape2.H, yshape2.W, yshape2.C, POOL_SIZE);
    
    free(y1);
    free(y2);

    return yshape;
}

Shape3 UnetUp(UnetUpParams* w, float* x, float* skip, float** y, int H, int W, int n_in, int n_out, int size_k, int size_p) {
    
    LayerParams* params_block0 = w->params_block0;    
    ResidualConvParams* params_block1 = w->params_block1;
    ResidualConvParams* params_block2 = w->params_block2; 

    float* weight = params_block0->weight;
    float* bias = params_block0->bias;

    int conv_trans_size_k = 2;
    int conv_trans_size_s = 2;
    
    float* x1 = concat_channel(x, skip, n_in/2, n_in/2, H, W);
    
    float* y1;
    Shape3 yshape1 = convTrans2d(weight, bias, x1, &y1, H, W, n_in, n_out, conv_trans_size_k, conv_trans_size_s);

    float* y2;
    Shape3 yshape2 = ResidualConvBlock(params_block1, y1, &y2, yshape1.H, yshape1.W, yshape1.C, n_out, size_k, size_p, false);

    Shape3 yshape = ResidualConvBlock(params_block2, y2, &(*y), yshape2.H, yshape2.W, yshape2.C, n_out, size_k, size_p, false);

    free(x1);
    free(y1);
    free(y2);

    Shape3 shape = yshape;
    return shape;
}



Shape3 ContextUnet(ContextUnetParams* w, float* x,  float* t, float* c, float** y, int H, int W, int n_in, int n_out, int n_cfeature, int size_k, int size_p, int up0_size_k, int up0_size_s) {
    ResidualConvParams* params_block0 = w->params_block0;
    UnetDownParams* params_block1 = w->params_block1;
    UnetDownParams* params_block2 = w->params_block2;
    EmbFCParams* timeembed1 = w->timeembed1;
    EmbFCParams* timeembed2 = w->timeembed2;
    EmbFCParams* contextembed1 = w->contextembed1;
    EmbFCParams* contextembed2 = w->contextembed2;
    LayerParams* params_up0 = w->params_up0;
    UnetUpParams* params_up1 = w-> params_up1;
    UnetUpParams* params_up2 = w-> params_up2;
    OutParams* params_out = w-> params_out;

    float* y1; 
    Shape3 shape_y1 = ResidualConvBlock(params_block0, x, &y1, H, W, n_in, n_out, size_k, size_p, true);

    float* down1;
    Shape3 shape_d1 = UnetDown(params_block1, y1, &down1, shape_y1.H, shape_y1.W, shape_y1.C, n_out, size_k, size_p);
    
    float* down2;
    Shape3 shape_d2 = UnetDown(params_block2, down1, &down2, shape_d1.H, shape_d1.W, shape_d1.C, 2*n_out, size_k, size_p);

    float* v;
    Shape3 shape_v = to_vec(down2, &v, shape_d2.H, shape_d2.W, shape_d2.C, 4);

    float* up0;
    Shape3 shape_up0 = Up0(params_up0, v, &up0, shape_v.H, shape_v.W, shape_v.C, shape_v.C, up0_size_k, up0_size_s);

    float* temb1;
    Shape1 size_temb1 = EmbFC(timeembed1, t, &temb1, 1, shape_up0.C);

    float* cemb1;
    Shape1 size_cemb1 = EmbFC(contextembed1, c, &cemb1, n_cfeature, shape_up0.C);

    elm_linear(up0, cemb1, temb1, shape_up0.C, shape_up0.H*shape_up0.W);

    float* up1;
    Shape3 shape_up1 = UnetUp(params_up1, up0, down2, &up1, shape_up0.H, shape_up0.W, shape_up0.C + shape_d2.C, n_out, size_k, size_p);
    
    float* temb2;
    Shape1 size_temb2 = EmbFC(timeembed2, t, &temb2, 1, shape_up1.C);

    float* cemb2;
    Shape1 size_cemb2 = EmbFC(contextembed2, c, &cemb2, n_cfeature, shape_up1.C);

    elm_linear(up1, cemb2, temb2, shape_up1.C, shape_up1.H*shape_up1.W);

    float* up2;
    Shape3 shape_up2 = UnetUp(params_up2, up1, down1, &up2, shape_up1.H, shape_up1.W, shape_up1.C + shape_d1.C, n_out, size_k, size_p);

    Shape3 shape_y = Out(params_out, up2, y1, &(*y), H, W, n_in, n_out, size_k, size_p); 


    free(y1);
    free(down1);
    free(down2);
    free(v);
    free(up0);
    free(temb1);
    free(cemb1);
    free(up1);
    free(temb2);
    free(cemb2);
    free(up2);

    return shape_y;

}


float* normalize(float* x, int C, int H, int W) {
    float min_value[C];
    float max_value[C];
    int SIZE = C*H*W;

    for (int c = 0; c < C; c++) {
        min_value[c] = 10000.0;
        max_value[c] = -10000.0;
    }

    // Find the min and max values for each channel
    for (int i = 0; i < SIZE; i++) {
        int c = i / (H * W);
        if (x[i] < min_value[c]) {
            min_value[c] = x[i];
        }
        if (x[i] > max_value[c]) {
            max_value[c] = x[i];
        }
    }
    float* y = calloc(SIZE, sizeof(float));
    // Normalize the 1D array based on its corresponding channel
    int i;
//    #pragma omp parallel private(i)
    for (i = 0; i < SIZE; i++) {
        int c = i / (H * W);
       y[i] = (x[i] - min_value[c]) / (max_value[c] - min_value[c]);
    }
    return y;
}

void printSquareWithColor(int r, int g, int b, int size) {
    // ANSI escape code for setting background color using RGB values
    printf("\033[48;2;%d;%d;%dm", r, g, b);

    // Print a block of spaces to represent the square
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size * 2; j++) {  // Multiply by 2 for width
            printf(" ");
        }
    }

    // Reset the background color to default
    printf("\033[0m");
}

void displayImageInTerminal(float *colors,  int height, int width) {
    int size = height * width;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            int r = (int)(colors[index]*255);
            int g = (int)(colors[index + size]*255);
            int b = (int)(colors[index + 2 * size]*255);
            printSquareWithColor(r, g, b, 1);
        }
        printf("\n");
    }
}


float randn() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;

    // Box-Muller transform
    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    // float z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2); // Second independent value if needed

    return z0;
}

void denoise_add_noise(float* x,  float* pred_noise, float a_t, float b_t, float ab_t, int size) {
  for(int i=0; i<size; i++) {
    float z = randn();
    float noise = sqrt(b_t)*z;
    float mean = (x[i] - pred_noise[i]*((1-a_t)/sqrt(1-ab_t)))/sqrt(a_t);
    x[i] = mean + noise;
  }
}

#pragma pack(push, 1)
typedef struct {
    short type;
    int file_size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct {
    int size;
    int width;
    int height;
    short planes;
    short bits_per_pixel;
    unsigned compression;
    unsigned image_size;
    int x_resolution;
    int y_resolution;
    int n_colors;
    int important_colors;
} BMPInfoHeader;
#pragma pack(pop)

void save_pixel_art_bmp(const char *filename, float *imageData, int height, int width) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "can't open %s\n", filename);
        return;
    }

    int padding = (4 - (width * 3) % 4) % 4;
    int row_size = (width * 3) + padding;
    int image_size = row_size * height;

    BMPHeader header = {
        .type = 0x4D42,
        .file_size = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + image_size,
        .reserved1 = 0,
        .reserved2 = 0,
        .offset = sizeof(BMPHeader) + sizeof(BMPInfoHeader)
    };

    BMPInfoHeader info_header = {
        .size = sizeof(BMPInfoHeader),
        .width = width,
        .height = height,
        .planes = 1,
        .bits_per_pixel = 24,
        .compression = 0,
        .image_size = image_size,
        .x_resolution = 0,
        .y_resolution = 0,
        .n_colors = 0,
        .important_colors = 0
    };

    fwrite(&header, sizeof(BMPHeader), 1, fp);
    fwrite(&info_header, sizeof(BMPInfoHeader), 1, fp);

    unsigned char *buffer = malloc(row_size * sizeof(unsigned char));
    if (!buffer) {
        fprintf(stderr, "Could not allocate buffer for BMP image data\n");
        fclose(fp);
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            buffer[x * 3]     = (unsigned char)(imageData[2 * height * width + y * width + x] * 255); // Blue
            buffer[x * 3 + 1] = (unsigned char)(imageData[height * width + y * width + x] * 255);   // Green
            buffer[x * 3 + 2] = (unsigned char)(imageData[y * width + x] * 255);          // Red
        }
        for (int p = 0; p < padding; p++) {
            buffer[width * 3 + p] = 0;
        }
        fwrite(buffer, sizeof(unsigned char), row_size, fp);
    }

    free(buffer);
    fclose(fp);
}



float* data = NULL; // memory mapped data pointer
ssize_t file_size;
Config cfg;
ContextUnetParams weights;


int load_model() {
    char* ckpt = "weights/ckpt.bin";
    int fd = 0; // file descriptor for memory mapping

    {
        FILE* file = fopen(ckpt, "rb");

        if (!file) { printf("Couldn't open file %s\n", ckpt); return 1; }
        if (fread(&cfg, sizeof(Config), 1, file) != 1) { return 1; }

        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file); // get the file size, in bytes

        printf("File size: %ld bytes\n", file_size);
        fclose(file);

        fd = open(ckpt, O_RDONLY); // open in read only mode

        if (fd == -1) { printf("open failed!\n"); return 1; }

        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

        if (data == MAP_FAILED) { printf("mmap failed!\n"); return 1; }
        float* weights_ptr = data + sizeof(Config)/sizeof(float);

        init_context_unet_weights(&weights, cfg.n_in, cfg.n_feature, cfg.n_cfeature, SIZE_K, cfg.height/4, cfg.height/4, weights_ptr);
    }

    if (fd != -1) close(fd);
    return 0;
}

int free_model(){
//    free(weights.params_block0->weight);
//    free(weights.params_block0->bias);
//    free(weights.params_block0->weight2);
//    free(weights.params_block0->bias2);
//    free(weights.params_block0->weight3);
//    free(weights.params_block0->bias3);
    free(weights.params_block0);
    free(weights.params_block1->params_block1);
    free(weights.params_block1->params_block2);
    free(weights.params_block1 );
    free(weights.params_block2->params_block1);
    free(weights.params_block2->params_block2);
    free(weights.params_block2);
    free(weights.timeembed1->params_block1);
    free(weights.timeembed1->params_block2);
    free(weights.timeembed1);
    free(weights.timeembed2->params_block1);
    free(weights.timeembed2->params_block2);
    free(weights.timeembed2);
    free(weights.contextembed1->params_block1);
    free(weights.contextembed1->params_block2);
    free(weights.contextembed1);
    free(weights.contextembed2->params_block1);
    free(weights.contextembed2->params_block2);
    free(weights.contextembed2);
    free(weights.params_up0);
    free(weights.params_up1->params_block0);
    free(weights.params_up1->params_block1);
    free(weights.params_up1->params_block2);
    free(weights.params_up1);
    free(weights.params_up2->params_block0);
    free(weights.params_up2->params_block1);
    free(weights.params_up2->params_block2);
    free(weights.params_up2);
    free(weights.params_out);
    if (data != MAP_FAILED) munmap(data, file_size);
}

int infer(int timesteps, float beta1, float beta2, int cidx){
        float* c = calloc(cfg.n_cfeature, sizeof(float));
        if(cidx < 0) {
          c[rand() % 5] = 1.0;
        } else {
          c[cidx % 5] = 1.0;
        }


        int size_x = cfg.n_in*cfg.height*cfg.width;
        float* x = malloc(size_x * sizeof(float));
        for(int i=0; i<size_x; i++) {
            x[i] = randn();
        }


        float* y; //predicted noise
        float* img;

        float b_t[timesteps + 1];
        float a_t[timesteps + 1];
        float ab_t[timesteps + 1];

        for (int i = 0; i <= timesteps; i++) {
            float t = (float)i/(float)timesteps; // equivalent to torch.linspace(0, 1, timesteps + 1)
            b_t[i] = (beta2 - beta1) * t + beta1;
            a_t[i] = 1 - b_t[i];
        }

        ab_t[0] = 1;
        for (int i = 1; i <= timesteps; i++) {
            ab_t[i] = ab_t[i-1] * a_t[i];
        }
        for(int i=timesteps; i>0; i--) {
            float* t = malloc(sizeof(float));
            t[0] = (float)i/(float)timesteps;

            float _a_t=a_t[i];
            float _b_t = b_t[i];
            float _ab_t = ab_t[i];
//            printf("t: %.4f, a_t: %.4f, b_t: %.4f, ab_t: %.4f\n", t[0], _a_t, _b_t, _ab_t);


            Shape3 yshape=ContextUnet(
              &weights, x, t, c, &y,
              cfg.height, cfg.width, cfg.n_in, cfg.n_feature, cfg.n_cfeature,
              SIZE_K, SIZE_P, cfg.height/4, cfg.height/4
            );

            denoise_add_noise(x, y, _a_t, _b_t, _ab_t, yshape.size);
            img = normalize(x, yshape.C, yshape.H, yshape.W);
            #ifdef _WIN32
                int r = system("cls");
            #else
                int r = system("clear");
            #endif
            displayImageInTerminal(img, yshape.H, yshape.W);
            printf("\n");

           switch (cidx + 1) {
              case 1:
                  printf("\033[32mgenerating a human icon, steps: %d/%d\033[0m\n", timesteps-i, timesteps);
                  break;
              case 2:
                  printf("\033[32mgenerating a non-human icon, steps: %d/%d\033[0m\n", timesteps-i, timesteps);
                  break;
              case 3:
                  printf("\033[32mgenerating a food icon, steps: %d/%d\033[0m\n", timesteps-i , timesteps);
                  break;
              case 4:
                  printf("\033[32mgenerating a spell icon, steps: %d/%d\033[0m\n", timesteps-i, timesteps);
                  break;
              case 5:
                  printf("\033[32mgenerating a side-facing icon, steps: %d/%d\033[0m\n", timesteps-i, timesteps);
                  break;
              default:
                  printf("\033[32mgenerating a random icon, steps: %d/%d\033[0m\n", timesteps-i, timesteps);
                  break;
          }

            free(t);
        }
        free(x);
        free(c);
        free(y);
        free(img);
        return 0;
}

int main() {
    srand(time(NULL));
    load_model();

    int choice;

    // Display the menu
    printf("input what you wanna generate:\n");
    printf("1. human icon\n");
    printf("2. non-human icon\n");
    printf("3. food icon\n");
    printf("4. spell icon\n");
    printf("5. side-facing icon\n");
    printf("Enter your choice (1/2/3/4/5), otherwise generate an icon of random type: ");

    // Read the user's choice
    int result = scanf("%d", &choice);
    // Process the choice
    int cidx;

    switch (choice) {
        case 1:
            infer(200, 1e-4, 0.02, 0);
            break;
        case 2:
            infer(200, 1e-4, 0.02, 1);
            break;
        case 3:
            infer(200, 1e-4, 0.02, 2);
            break;
        case 4:
            infer(200, 1e-4, 0.02, 3);
            break;
        case 5:
            infer(200, 1e-4, 0.02, 4);
            break;
        default:
            infer(200, 1e-4, 0.02, -1);
            break;
    }
//    time_t start = time(NULL);
//    infer();


//    time_t end = time(NULL);
//    float elapsed_time = difftime(end, start);
//    printf("Time taken: %f seconds\n", elapsed_time);

    free_model();
    return 0;
}