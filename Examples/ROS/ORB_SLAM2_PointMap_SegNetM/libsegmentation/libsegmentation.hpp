/*
 *--------------------------------------------------------------------------------------------------
 * An FPGA Based Energy Efficient DS-SLAM Accelerator for Mobile Robots in Dynamic Environment
 * Author(s):
 * Yakun Wu, Li Luo, Shujuan Yin, Mengqi Yu, Hongzhi Huang, Xuesong Shi, Qi Wei, Xinjun Liu and Fei Qiao qiaofei@mail.tsinghua.edu.cn
 * Created by Yakun Wu@2021.06.20
 * -------------------------------------------------------------------------------------------------- 
 * An energy-efficient DS-SLAM based semantic SLAM system is proposed on the HERO heterogeneous platform. 
 * This work is based on DS-SLAM system. If you haven't learn DS-SLAM code, 
 * you'd better to be familiar with DS-SLAM project first. Compared to DS-SLAM, 
 * You should pay attention to libsegmentation.
 *
 * --------------------------------------------------------------------------------------------------
 * Copyright (C) 2021, iVip Lab @EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/). All rights reserved.
 *
 * Licensed under the GPLv3 License;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/ivipsourcecode/DS-SLAM/blob/master/LICENSE
 *--------------------------------------------------------------------------------------------------
 */

#ifndef LIBSEGMENTATION_HPP  
#define LIBSEGMENTATION_HPP 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

// CNN network configuration file
#include "./device/segnet_hw_param666_W.cl"

#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

// SW System parameters
#define DMA_ALIGNMENT 64
#define MAX_LAYER_NUM 26
#define MAX_BATCH_SIZE 1

#define IN_BUF_SIZE 256 * 256 * 128 // Note: the buffer size should be large enough to hold all temperary results
#define LOCATION_BUF_SIZE1 128 * 128 * 128 * MAX_BATCH_SIZE
#define LOCATION_BUF_SIZE2 64 * 64 * 256 * MAX_BATCH_SIZE
#define LOCATION_BUF_SIZE3 32 * 32 * 512 * MAX_BATCH_SIZE
#define LOCATION_BUF_SIZE4 16 * 16 * 512 * MAX_BATCH_SIZE

#define IMAGE_FILE_SIZE   (224*224*3)//mean!!!
#define TEST_DATA_WIDTH 224
#define TEST_DATA_HEIGHT 224
#define TEST_DATA_CHANNEL 21
#define DATA_CHANNEL 3
#define PICTURE_NUM 1
#define MAX_PIC_NUM 1436

#define WEIGHTS_FILE_SIZE 29439253
#define FRAC_FILE_SIZE 15914
#define LAYER_NUM 26
#define CONV_NUM 26
//mean
#define MEAN_DATA_WIDTH   224
#define MEAN_DATA_HEIGHT  224
#define MEAN_DATA_CHANNEl 3

typedef signed char DTYPE;

using namespace cv;
using namespace std;
using namespace ocl_util;

class Classifier
{

    public:
    Classifier(const string& mean_file,
    const string& weights_file,
    const string& frac_file,
    const string& aocx_file);

    const string& mean_file;
    const string& weights_file;
    const string& frac_file;
    const string& aocx_file;

    void fpgainit();
    cv::Mat Predict(const cv::Mat& img,  cv::Mat LUT_image);
  
    // private:
    void loadImageToBuffer(const cv::Mat& img);
    int  prepare();
    void readDataBack();
    void verifyResult();
    void reorderWeights(DTYPE *weights, DTYPE *weight_buf, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim3_original, unsigned dim4_original, 											unsigned offset, unsigned padding_offset, unsigned vecSize, unsigned laneNum);
    void reorderBias(DTYPE *dataIn, DTYPE *bias, unsigned offset, unsigned padding_offset, unsigned dim4, unsigned dim4_original, unsigned laneNum);
    void reorderOutput(DTYPE *output, DTYPE *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3, unsigned padding_top, unsigned padding_bottom);
    void cleanupall();
    
    //------------ Global Functions & Variables ------------//
    cl_uint num_devices;
    cl_platform_id platform_id;
    cl_context context;
    cl_program program;
    
    const char *knl_name_memRd = "memRead";
    const char *knl_name_locationRd = "locationRd";
    const char *knl_name_conv = "coreConv";
    const char *knl_name_pool = "maxPool";
    const char *knl_name_unpooling = "unpooling";
    const char *knl_name_memWr = "memWrite";

    scoped_array<cl_device_id> device;

    scoped_array<cl_kernel> knl_memRd;
    scoped_array<cl_kernel> knl_locationRd;
    scoped_array<cl_kernel> knl_conv;
    scoped_array<cl_kernel> knl_pool;
    scoped_array<cl_kernel> knl_unpooling;
    scoped_array<cl_kernel> knl_memWr;

    scoped_array<cl_command_queue> que_memRd;
    scoped_array<cl_command_queue> que_locationRd;
    scoped_array<cl_command_queue> que_conv;
    scoped_array<cl_command_queue> que_pool;
    scoped_array<cl_command_queue> que_unpooling;
    scoped_array<cl_command_queue> que_memWr;

    scoped_array<cl_mem> data_buf;
    scoped_array<cl_mem> output_buf;
    scoped_array<cl_mem> weights_buf;
    scoped_array<cl_mem> bias_buf;
    scoped_array<cl_mem> frac_bias_buf;
    scoped_array<cl_mem> frac_weight_buf;
    scoped_array<cl_mem> location_buf1;
    scoped_array<cl_mem> location_buf2;
    scoped_array<cl_mem> location_buf3;
    scoped_array<cl_mem> location_buf4;//!!!

    // typedef signed char DTYPE;
    const char *vendor_name = "Intel";
    DTYPE *image;//mean!!!
    DTYPE *weights;
    DTYPE *frac;
    DTYPE *data_init;
    float *mean_data;
    DTYPE *weight_conv[MAX_LAYER_NUM];
    DTYPE *bias_conv[MAX_LAYER_NUM];
    DTYPE *frac_bias_conv[MAX_LAYER_NUM];
    DTYPE *frac_weight_conv[MAX_LAYER_NUM];
    DTYPE *output;
    DTYPE *output_reorder;

    int ind[TEST_DATA_HEIGHT][TEST_DATA_WIDTH];

    cl_ulong kernel_time[LAYER_NUM][6]; //???
    cl_ulong max_kernel_time[LAYER_NUM];
};

#endif
