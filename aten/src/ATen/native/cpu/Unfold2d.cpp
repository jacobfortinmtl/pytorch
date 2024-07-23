#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Unfold2d.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/utils.h>
#include <cmath>
#include <iostream>
#include <cstdlib> // For std::getenv
#include <string>  // For std::stoi
#include <iostream> // For std::cout
#include <chrono>

namespace at::native {

namespace {

template <typename scalar_t>
static inline void cadd(
    scalar_t* z,
    const scalar_t* x,
    const scalar_t* y,
    int64_t n) {
  using Vec = vec::Vectorized<scalar_t>;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  char* ptrs[] = {reinterpret_cast<char*>(z),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(x)),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(y))};
  vectorized_loop(
      ptrs,
      n,
      -1,
      [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },
      [](Vec x, Vec y) -> Vec { return x + y; });
}

template <typename scalar_t>
static void unfolded2d_acc(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(0, n_input_plane, 0, [&](int64_t start, int64_t end) {
    for (const auto nip : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t kw, kh, y, x;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t ix, iy;
      for (kh = 0; kh < kH; kh++) {
        for (kw = 0; kw < kW; kw++) {
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* dst =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t lpad, rpad;
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = std::max<int64_t>(0, padW - kw);
                  rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + lpad;
                  cadd(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width + lpad,
                      output_width - lpad - rpad);
                } else {
                  for (x = 0; x < output_width; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width) {
                    } else {
                      scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                      *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                    }
                  }
                }
              }
            }
          } else {
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1) {
                scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                cadd(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * output_width,
                    output_width);
              } else {
                for (x = 0; x < output_width; x++) {
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + x * dW;
                  *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void unfolded2d_acc_channels_last(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {

  for (int64_t y = 0; y < output_height; y++) {
    for (int64_t x = 0; x < output_width; x++) {
      scalar_t* src = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      scalar_t* dst = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
            } else {
              scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
              scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
              cadd(dst_slice,
                   dst_slice,
                   src_slice,
                   n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
            scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
            cadd(dst_slice,
                 dst_slice,
                 src_slice,
                 n_input_plane);
          }
        }
      }
    }
  }
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded2d_copy */
void unfolded2d_acc_kernel(
    ScalarType dtype,
    void *finput_data,
    void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc_channels_last", [&] {
      unfolded2d_acc_channels_last(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
     });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc", [&] {
      unfolded2d_acc(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
      });
  }
}

template <typename scalar_t>
static void unfolded2d_copy(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH, // Kernel Height size
    int64_t kW, 
    int64_t dH, // Stride
    int64_t dW,
    int64_t padH, 
    int64_t padW,
    int64_t n_input_plane, // Channels
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    // Getting an env variable to check for for folding
    // starting time calculation
    
    int flag = 1;
    char* env_var = std::getenv("DEFAULT");
    char* env_var_conv = std::getenv("CONVERSION");
    if (env_var != NULL && std::string(env_var) == "1") {
      flag = 0;
    }
    if (env_var_conv != NULL && std::string(env_var_conv) == "1") {
      flag = 0;
    }
    if (flag == 1){
      auto start = std::chrono::high_resolution_clock::now();
      // std::cout << "Using row major order for im2col" << std::endl;
      std::cout << std::endl;
      const scalar_t* src = input_data;
      scalar_t* dest = finput_data;
      /* Option to create the memory storage of im2col in row major order.
      Not every row in the input will appear in windows the same amount of times.
      Namely, given a N x N matrix, and kernel of k x k,  
      rows from r = 1 to r = n - k + 1 will appear n - k + 1 times
      */
      // //printing destination after first loop
      // std::cout << "Unfolded data (flat): \n" << std::endl;
      // int64_t total_size = n_input_plane * kH * kW * output_height * output_width;
      // for (int i = 0; i < total_size; i++) {
      //   std::cout << dest_start[i] << " ";
      // }
      // // Priting what value all the pointers in new_index are pointing at
      // for (int i = 0; i < output_height * output_width; i++) {
      //   std::cout << "Index " << i << " is pointing at: " << *(new_index[i]-1) << std::endl;
      // }
      int offset = 0;
      
      for (int rowsKernel = 0; rowsKernel < kH; rowsKernel++) { //start at first row since row one started above
        for (int ip = 0; ip < n_input_plane; ip++) { // looping through channels
          #pragma omp parallel for private (src, dest) collapse(2)
          for (int ow = 0; ow < output_width; ow++) {
            for(int oh = 0; oh < input_height; oh++) {
              // Checking if current row in input can be a top row of a window
              // e.g., with a 2x2 kernel, the bottom row cannot be the top row of a window
              if (oh - rowsKernel >= output_height) {
                continue;
              } 
              // Checking it current row in input can be a bottom row of a window, should
              // fail for the first row
              else if (oh - rowsKernel < 0) { 
                continue;
              }
              offset = rowsKernel*(output_width*kH*kW - kW); //every row needs to be offset to the left
              src = 
                input_data + ip * input_height * input_width + oh * input_width + ow;
              dest = 
                (finput_data + 
                ip * kH * kW * output_height * output_width + // not sure if channels work
                ow * kH*kH + oh * kH * kW * output_width - offset);
              // Printing the current indexing of rowskernel, ow, oh
              // std::cout << "Indexing: Rows: " << rowsKernel << ", oh:" << oh << ", OW: " << ow << std::endl;
              // // Printing the current src and dest pointers
              // std::cout << "Src: " << ip * input_height * input_width + oh * input_width + ow 
              // << ", Dest: " << ip * kH * kW * output_height * output_width + // not sure if channels work
              //   ow * kH*kH + oh * kH * kW * output_width - offset << std::endl;
              // std::cout << "Offset: " << offset << std::endl;
              memcpy(dest, src, kW * sizeof(scalar_t));
            }
          }
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for im2ROW: " << elapsed.count() << std::endl;
    }
    else{
        auto start = std::chrono::high_resolution_clock::now();
        at::parallel_for(
        0, (int64_t)n_input_plane * kH * kW, 0, [&](int64_t start, int64_t end) {
          for (const auto k : c10::irange(start, end)) {
            // these are indices not sizes for the flattened input!!!!!
            int64_t nip = k / (kH * kW);
            int64_t rest = k % (kH * kW);
            int64_t kh = rest / kW;
            int64_t kw = rest % kW;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t x, y;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t ix, iy;
            scalar_t* dst = finput_data +
                nip * ((size_t)kH * kW * output_height * output_width) +
                kh * ((size_t)kW * output_height * output_width) +
                kw * ((size_t)output_height * output_width);
            const scalar_t* src =
                input_data + nip * ((size_t)input_height * input_width);
            if (padW > 0 || padH > 0) {
              // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
              int64_t lpad, rpad;
              for (y = 0; y < output_height; y++) {
                iy = (int64_t)y * dH - padH + kh;
                if (iy < 0 || iy >= input_height) {
                  memset(
                      dst + (size_t)y * output_width,
                      0,
                      sizeof(scalar_t) * output_width);
                } else {
                  if (dW == 1) {
                    ix = 0 - padW + kw;
                    lpad = std::max<int64_t>(0, padW - kw);
                    rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                    if (output_width - rpad - lpad <= 0) {
                      memset(
                          dst + (size_t)y * output_width,
                          0,
                          sizeof(scalar_t) * output_width);
                    } else {
                      if (lpad > 0)
                        memset(
                            dst + (size_t)y * output_width,
                            0,
                            sizeof(scalar_t) * lpad);
                      memcpy(
                          dst + (size_t)y * output_width + lpad,
                          src + (size_t)iy * input_width + ix + lpad,
                          sizeof(scalar_t) * (output_width - rpad - lpad));
                      if (rpad > 0)
                        memset(
                            dst + (size_t)y * output_width + output_width - rpad,
                            0,
                            sizeof(scalar_t) * rpad);
                    }
                  } else {
                    for (x = 0; x < output_width; x++) {
                      ix = (int64_t)x * dW - padW + kw;
                      if (ix < 0 || ix >= input_width)
                        memset(
                            dst + (size_t)y * output_width + x,
                            0,
                            sizeof(scalar_t) * 1);
                      else
                        memcpy(
                            dst + (size_t)y * output_width + x,
                            src + (size_t)iy * input_width + ix,
                            sizeof(scalar_t) * (1));
                    }
                  }
                }
              }
            } else {
              for (y = 0; y < output_height; y++) {
                iy = (int64_t)y * dH + kh;
                ix = 0 + kw;
                if (dW == 1) {
                    // std::cout << "Copying row from (" << iy << ", " << ix << "): ";
                    // for (int i = 0; i < output_width; i++) {
                    //     std::cout << *(src + (size_t)iy * input_width + ix + i) << " ";
                    // }
                    // std::cout << std::endl;
                    memcpy(
                        dst + (size_t)y * output_width,
                        src + (size_t)iy * input_width + ix,
                        sizeof(scalar_t) * output_width);
                }
                else {
                  for (x = 0; x < output_width; x++) {
                  // std::cout << "Copying element from (" << iy << ", " << (ix + x * dW) << "): ";
                  // std::cout << *(src + (size_t)iy * input_width + ix + (int64_t)x * dW) << std::endl;
                  memcpy(
                      dst + (size_t)y * output_width + x,
                      src + (size_t)iy * input_width + ix + (int64_t)x * dW,
                      sizeof(scalar_t) * (1));
                  }
                }
              }
            }
          }
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for window to columns: " << elapsed.count() << std::endl;
  }
  
    // printing unfolded data
    // std::cout << "Unfolded data (flat): \n" << std::endl;
    // int64_t total_size = n_input_plane * kH * kW * output_height * output_width;
    // for (int i = 0; i < total_size; i++) {
    //   std::cout << finput_data[i] << " ";
    // }
    // std::cout << std::endl;
    // // Printing col major order
    // int height_col_size = output_height*output_width;
    // for (int i = 0; i < height_col_size; i++) { //m = num rows A
    //     std::cout << "Row " << i << ": ";
    //     for (int j = 0; j < output_width; j++) { //k = common dimension
    //         int index = j * height_col_size + i;
    //         std::cout << finput_data[index] << " ";
    //     }
    //     std::cout << std::endl;
    // } 
}

template <typename scalar_t>
static void unfolded2d_copy_channels_last(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(0, output_height * output_width, 0, [&](int64_t start, int64_t end) {
    int64_t y = 0;
    int64_t x = 0;
    data_index_init(start, y, output_height, x, output_width);

    for (const auto k C10_UNUSED: c10::irange(start, end)) {
      scalar_t* dst = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      const scalar_t* src = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
              memset(dst + kh * kW * n_input_plane + kw * n_input_plane,
                    0,
                    sizeof(scalar_t) * n_input_plane);
            } else {
              memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                     src + iy * input_width * n_input_plane + ix * n_input_plane,
                     sizeof(scalar_t) * n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                   src + iy * input_width * n_input_plane + ix * n_input_plane,
                   sizeof(scalar_t) * n_input_plane);
          }
        }
      }
      // move on to next output index
      data_index_step(y, output_height, x, output_width);
    }
  });
}

void unfolded2d_copy_kernel(
    ScalarType dtype,
    void *finput_data,
    const void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // kH*kW does not overflow an int
  // n_input_plane*kH*kW does not overflow a int64_t
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy_channels_last", [&] {
      unfolded2d_copy_channels_last(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy", [&] {
      unfolded2d_copy(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  }
}

} // namespace

REGISTER_DISPATCH(unfolded2d_copy_stub, &unfolded2d_copy_kernel);
REGISTER_DISPATCH(unfolded2d_acc_stub, &unfolded2d_acc_kernel);

} // namespace at::native
