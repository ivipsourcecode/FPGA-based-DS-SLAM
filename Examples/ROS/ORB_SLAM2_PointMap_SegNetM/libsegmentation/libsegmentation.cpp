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

#include "libsegmentation.hpp"
#include <chrono> 
#include "segnet_pascal_layer_config.h"
 
using namespace cv;
using namespace std;
using namespace ocl_util;

enum config_item {data_w,data_h,data_n,weight_w,weight_h,weight_n,weight_m,bias_size,frac_bias_size,frac_weight_size, // memRd pram
memrd_src, //"0"-> data_buf  "1"-> output_buf
locationRd_src, //"0"-> Don't read location  "1"-> location_buf1  "2"->// location_buf2 "3"-> location_buf3
conv_x,conv_y,conv_z,conv_stride,conv_padding,conv_split,relu_on, // Conv Parameters
pool_on,pool_x,pool_y,pool_z,pool_size,pool_stride, // Pool Parameters
unpooling_on,unpooling_x,unpooling_y,unpooling_z,unpooling_odd_flag_x,unpooling_odd_flag_y, // Unpooling Parameters
memwr_dst, //"0"-> data_buf  "1"-> output_buf
locationWr_dst //"0"-> Don't write location  "1"-> location_buf1  "2"->// location_buf2 "3"-> location_buf3
};
enum input_item {image_w,image_h,image_n, // original image size
batch_size
};
enum output_item {output_w,output_h,output_n};

enum precision_item {frac_din, frac_dout};

unsigned layer_config_original[LAYER_NUM][NUM_CONFIG_ITEM];

//typedef signed char DTYPE;
//const char *vendor_name = "Intel";

class Classifier;

Classifier::Classifier(const string& mean_file,
                       const string& weights_file,
                       const string& frac_file,
                       const string& aocx_file):mean_file(mean_file),weights_file(weights_file),frac_file(frac_file),aocx_file(aocx_file)
{
	//------------ Global Functions & Variables ------------//
	num_devices = 0;
	platform_id = NULL;
	context = NULL;
	program = NULL;
	int ind[TEST_DATA_HEIGHT][TEST_DATA_WIDTH] = {0};
}
void Classifier::fpgainit() {

	cl_int status;
  	unsigned int weight_buf_size;

	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if (platform_id == NULL) {
		printf("ERROR: Unable to find the desired OpenCL platform.\n");
		exit(0);
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	printf("num_devices: %d\n",num_devices);//!!!!!
	for (unsigned char i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}
	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");
	if(status != CL_SUCCESS) cleanupall();
	// Create Program Objects
	const char *kernel_file_name = &aocx_file[0];

	// Create the program for all device. All devices execute the same kernel.
	program = createProgramFromFile(context, kernel_file_name, device, num_devices);		
	
	// Create per-device objects.
	que_memRd.reset(num_devices);
	que_locationRd.reset(num_devices);
	que_conv.reset(num_devices);
	que_pool.reset(num_devices);
	que_unpooling.reset(num_devices);
	que_memWr.reset(num_devices);

	knl_memRd.reset(num_devices);
	knl_locationRd.reset(num_devices);
	knl_conv.reset(num_devices);
	knl_pool.reset(num_devices);
	knl_unpooling.reset(num_devices);
	knl_memWr.reset(num_devices);

	data_buf.reset(num_devices * MAX_BATCH_SIZE);
	output_buf.reset(num_devices * MAX_BATCH_SIZE);
	weights_buf.reset(num_devices * LAYER_NUM);
	bias_buf.reset(num_devices * LAYER_NUM);
	frac_bias_buf.reset(num_devices * LAYER_NUM);
	frac_weight_buf.reset(num_devices * LAYER_NUM);
	location_buf1.reset(num_devices * MAX_BATCH_SIZE);
	location_buf2.reset(num_devices * MAX_BATCH_SIZE);
	location_buf3.reset(num_devices * MAX_BATCH_SIZE);
	location_buf4.reset(num_devices * MAX_BATCH_SIZE);//!!!

	//!!! Prepare compute data
	status = prepare();
	if (status) {
		printf("Allocate memory for data and weights failed !!!\n");
		exit(0);
	}

	// Create qeues, kernels and mem objs
	for (unsigned char i = 0; i < num_devices; ++i) {
		// Command queue
		que_memRd[i] = clCreateCommandQueue(context, device[i],CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 0");
		if(status != CL_SUCCESS) cleanupall();
		que_locationRd[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 1");
		if(status != CL_SUCCESS) cleanupall();
		que_conv[i] = clCreateCommandQueue(context, device[i],CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 2");
		if(status != CL_SUCCESS) cleanupall();
		que_pool[i] = clCreateCommandQueue(context, device[i],CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 3");
		if(status != CL_SUCCESS) cleanupall();
		que_unpooling[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 4");
		if(status != CL_SUCCESS) cleanupall();
		que_memWr[i] = clCreateCommandQueue(context, device[i],CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 5");
		if(status != CL_SUCCESS) cleanupall();

		// Kernel
		knl_memRd[i] = clCreateKernel(program, knl_name_memRd, &status);
		checkError(status, "Failed to create memRd kernel");
		if(status != CL_SUCCESS) cleanupall();
		knl_locationRd[i] = clCreateKernel(program, knl_name_locationRd, &status);
		checkError(status, "Failed to create locationRd kernel");
		if(status != CL_SUCCESS) cleanupall();
		knl_conv[i] = clCreateKernel(program, knl_name_conv, &status);
		checkError(status, "Failed to create conv kernel");
		if(status != CL_SUCCESS) cleanupall();
		knl_pool[i] = clCreateKernel(program, knl_name_pool, &status);
		checkError(status, "Failed to create pool kernel");
		if(status != CL_SUCCESS) cleanupall();
		knl_unpooling[i] = clCreateKernel(program, knl_name_unpooling, &status);
		checkError(status, "Failed to create unpooling kernel");
		if(status != CL_SUCCESS) cleanupall();
		knl_memWr[i] = clCreateKernel(program, knl_name_memWr, &status);
		checkError(status, "Failed to create memWr kernel");
		if(status != CL_SUCCESS) cleanupall();

		// Create and initialize data buffers for each batch item
		for (unsigned char j = 0; j < LAYER_NUM; ++j) {

			weight_buf_size = layer_config[j][weight_w] * layer_config[j][weight_h] * layer_config[j][weight_n] * layer_config[j][weight_m];

			//// Create weight, bias, frac_bias and frac_weight buffers for each layer
			////// Weights buffers for each layer
			weights_buf[i * LAYER_NUM + j] = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_buf_size * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for weights in layer");
			if(status != CL_SUCCESS) cleanupall();
			////// Bias buffers for each layer
			bias_buf[i * LAYER_NUM + j] = clCreateBuffer(context, CL_MEM_READ_ONLY, layer_config[j][bias_size] * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for bias in layer");
			if(status != CL_SUCCESS) cleanupall();
			////// Frac_bias buffers for each layer
			frac_bias_buf[i * LAYER_NUM + j] = clCreateBuffer(context, CL_MEM_READ_ONLY, layer_config[j][frac_bias_size] * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for frac_bias in layer");
			if(status != CL_SUCCESS) cleanupall();
			////// frac_weight buffers for each layer
			frac_weight_buf[i * LAYER_NUM + j] = clCreateBuffer(context, CL_MEM_READ_ONLY, layer_config[j][frac_weight_size] * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for frac_weight in layer");
			if(status != CL_SUCCESS) cleanupall();

			//// Initializing weight, bias, frac_bias and frac_weight buffers,
			/// blocking write is used
			////// Weights buffers for each layer
			status = clEnqueueWriteBuffer(que_memRd[i], weights_buf[i*LAYER_NUM+j], CL_TRUE, 0, weight_buf_size * sizeof(DTYPE), weight_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer weight");
			if(status != CL_SUCCESS) cleanupall();
			////// Bias buffers for each layer
			status = clEnqueueWriteBuffer(que_memRd[i], bias_buf[i*LAYER_NUM+j], CL_TRUE, 0, layer_config[j][bias_size] * sizeof(DTYPE), bias_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer bias");
			if(status != CL_SUCCESS) cleanupall();
			////// Frac_bias buffers for each layer
			status = clEnqueueWriteBuffer(que_memRd[i], frac_bias_buf[i*LAYER_NUM+j], CL_TRUE, 0, layer_config[j][frac_bias_size] * sizeof(DTYPE), frac_bias_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer frac_bias");
			if(status != CL_SUCCESS) cleanupall();
			////// frac_weight buffers for each layer
			status = clEnqueueWriteBuffer(que_memRd[i], frac_weight_buf[i * LAYER_NUM + j], CL_TRUE, 0, layer_config[j][frac_weight_size] * sizeof(DTYPE), frac_weight_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer frac_weight");
			if(status != CL_SUCCESS) cleanupall();
		}
		for (unsigned char j = 0; j < input_config[batch_size]; ++j) {
			//// Create data buffers for each layer
			////// Input data buffers
			data_buf[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for data in layer");
			if(status != CL_SUCCESS) cleanupall();
			////// Output results buffers
			output_buf[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, IN_BUF_SIZE * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer for output in layer");
			if(status != CL_SUCCESS) cleanupall();

			// Create location buffers the whole net
			//// Location results buffer1
			location_buf1[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, LOCATION_BUF_SIZE1 * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer1 for location");
			if(status != CL_SUCCESS) cleanupall();
			//// Location results buffer2
			location_buf2[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, LOCATION_BUF_SIZE2 * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer2 for location");
			if(status != CL_SUCCESS) cleanupall();
			//// Location results buffer1
			location_buf3[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, LOCATION_BUF_SIZE3 * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer3 for location");
			if(status != CL_SUCCESS) cleanupall();
			//// !!!Location results buffer4
			location_buf4[i * input_config[batch_size] + j] = clCreateBuffer(context, CL_MEM_READ_WRITE, LOCATION_BUF_SIZE4 * sizeof(DTYPE), NULL, &status);
			checkError(status, "Failed to create buffer4 for location");
			if(status != CL_SUCCESS) cleanupall();
		}
  	}
}
cv::Mat Classifier::Predict(const cv::Mat& img, cv::Mat LUT_image) {   
	cl_int status;
    unsigned int	data_dim1xdim2;
    unsigned int	weight_dim4_div_lane;
    unsigned int	weight_dim1x2;
	unsigned int	weight_dim2x3;
    unsigned int 	weight_dim1x2x3;
    unsigned int 	group_num_x;
    unsigned int 	group_num_y;
	unsigned char 	group_rem_x;		
	unsigned int	group_rem_size_x_rd;
    unsigned int	group_rem_size_x;
    unsigned int	group_rem_size_y;
    unsigned int  	group_rem_size_xyz;
    unsigned int  	group_rem_size_xyz_out;		
	unsigned char 	win_gp_x;
    unsigned int	win_size_x;
    unsigned int	win_size_y;
    unsigned int  	win_size_xyz;
	unsigned int  	win_size_xyz_out;
    unsigned int  	location_group_num;
    unsigned int  	output_num;
    unsigned int  	conv_loop_cnt;
    unsigned int  	input_num;
    unsigned short 	line_size;
    unsigned short 	odd_flag_x;
    unsigned short 	odd_flag_y;
    unsigned int 	unpooling_outnum;
	unsigned int 	unpooling_dim3;
    unsigned short 	memWr_dim1, memWr_dim2;
    unsigned short 	memWr_dim3;
    unsigned short 	out_dim1xbatch;
    unsigned int  	out_dim1x2xbatch;
    unsigned char 	batch_indx_dim1;
    unsigned char 	batch_indx_dim2;
    unsigned char 	padding_offset_bottom;
    unsigned char 	padding_offset_top;

	unsigned iter_num;
	unsigned char batch_size_in_dim;

	//unsigned int weight_buf_size;
	unsigned char argi;
	unsigned int control;

	unsigned int pic_num = 1;

	size_t knl_memWr_global_size[3];
	size_t knl_memWr_local_size[3];

	// Recorde the excution time of each operation for each layer
	cl_ulong memRd_time[LAYER_NUM];
	cl_ulong locationRd_time[LAYER_NUM];
	cl_ulong conv_time[LAYER_NUM];
	cl_ulong pool_time[LAYER_NUM];
	cl_ulong unpooling_time[LAYER_NUM];
	cl_ulong memWr_time[LAYER_NUM];	
	
	// Execute the kernel
	scoped_array<cl_event> memRd_event(num_devices);
	scoped_array<cl_event> locationRd_event(num_devices);
	scoped_array<cl_event> conv_event(num_devices);
	scoped_array<cl_event> pool_event(num_devices);
	scoped_array<cl_event> unpooling_event(num_devices);
	scoped_array<cl_event> memWr_event(num_devices);
	
	Timer t; // Timer used for performance measurement
	float time;
	for (unsigned char i = 0; i < num_devices; ++i) {
		// Run Seg-PipeNN for multiple input pictures
		loadImageToBuffer(img);
		// Recorde the start time
		t.start();
		// Eexcute one pipe-line for layers:
		// MemRd ---> locationRd(?) ---> Conv(Bn(?) & Relu(?)) ---> Pool(?) ---> Unpooling(?) ---> MemWr
		for (unsigned char j = 0; j < LAYER_NUM; ++j) {
			memRd_time[j] = 0;
			locationRd_time[j] = 0;
			conv_time[j] = 0;
			pool_time[j] = 0;
			unpooling_time[j] = 0;
			memWr_time[j] = 0;

      		iter_num = input_config[batch_size]; // For conv layers, process by batch_size time

			// Each iteration process one item in batch
			for (unsigned char k = 0; k < iter_num; ++k) {
				// "0"-> bypass  "1"-> pool  "2"->  unpooling "3"-> pool=>>unpooling
				control = (layer_config[j][pool_on] & 0x01) | ((layer_config[j][unpooling_on] & 0x01) << 1); 										
				// Set Arguments
				////**********************************************************
				//// Set knl_memRd arguments
				////**********************************************************
				argi = 0;
				////// Convolution tasks (conv_x,conv_y) are divided into multiple groups
				group_num_x = ceil((float)layer_config[j][conv_x]/CONV_GP_SIZE_X);
				group_num_y = ceil((float)layer_config[j][conv_y]/CONV_GP_SIZE_Y);

				if(layer_config[j][conv_x]==1) {
					win_gp_x = 1;
					group_rem_x	= 1;
				}
				else{
					win_gp_x = CONV_GP_SIZE_X;
					if(layer_config[j][conv_x]%CONV_GP_SIZE_X==0)
						group_rem_x = CONV_GP_SIZE_X;
					else
						group_rem_x = layer_config[j][conv_x]%CONV_GP_SIZE_X;
				}

				weight_dim4_div_lane = layer_config[j][weight_m]/LANE_NUM
				weight_dim1x2 = layer_config[j][weight_w]*layer_config[j][weight_h];
				weight_dim2x3 = layer_config[j][weight_h]*layer_config[j][weight_n];
				weight_dim1x2x3	= layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n];
				win_size_x = layer_config[j][weight_w]+(win_gp_x-1)*layer_config[j][conv_stride];
				win_size_y = CONV_GP_SIZE_Y*layer_config[j][weight_h];
                win_size_xyz = win_size_x*win_size_y*layer_config[j][weight_n];
				win_size_xyz_out = weight_dim2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X;
				group_rem_size_x_rd	= layer_config[j][weight_w]+(group_rem_x-1)*layer_config[j][conv_stride];
				group_rem_size_x = group_rem_x*layer_config[j][weight_w];
            	group_rem_size_y = CONV_GP_SIZE_Y*layer_config[j][weight_h];
                group_rem_size_xyz_out = group_rem_size_x*group_rem_size_y*layer_config[j][weight_n]/WEIGHT_W;
				group_rem_size_xyz = (layer_config[j][weight_w]+(group_rem_x-1)*layer_config[j][conv_stride])*group_rem_size_y*layer_config[j][weight_n];		
                data_dim1xdim2 = layer_config[j][data_w]*layer_config[j][data_h];

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &layer_config[j][data_w]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &layer_config[j][data_h]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &data_dim1xdim2);
				checkError(status, "Failed to set rgument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_w]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_h]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &layer_config[j][weight_n]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &weight_dim4_div_lane);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &weight_dim2x3);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint),  &weight_dim1x2x3);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &layer_config[j][conv_x]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_stride]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_padding]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_split]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_num_x);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_num_y);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &group_rem_x);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_rem_size_x_rd);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_rem_size_x);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_rem_size_xyz);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &group_rem_size_xyz_out);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
								
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &win_size_x);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &win_size_y);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &win_size_xyz);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uint), &win_size_xyz_out);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
				
				////// Select the kernel input mem object source, used as ping-pong buffers
				////// data_buf -> conv -> output_buf -> conv -> data_buf -> ...
				if(layer_config[j][memrd_src]==0){
					status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
				}
				else {
					status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
				}	

				////**********************************************************
				//// Set knl_locationRd arguments
				////**********************************************************
				if ((control & 0x03) == 0x02) {
					argi = 0;
					location_group_num = ((layer_config[j][conv_y]*layer_config[j][conv_z]/LANE_NUM));

					status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_ushort), &layer_config[j][conv_x]);
					checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);

					status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_uint), &location_group_num);
					checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);

					if(layer_config[j][locationRd_src]==1){
						status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_mem), &location_buf1[i*input_config[batch_size]+k]);
						checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);
					}
					else if(layer_config[j][locationRd_src]==2){
						status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_mem), &location_buf2[i*input_config[batch_size]+k]);
						checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);
					}
					else if(layer_config[j][locationRd_src]==3){
						status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_mem), &location_buf3[i*input_config[batch_size]+k]);
						checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);
					}
					else if(layer_config[j][locationRd_src]==4){
						status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_mem), &location_buf4[i*input_config[batch_size]+k]);
						checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);
					}
					else{
						status = clSetKernelArg(knl_locationRd[i], argi++, sizeof(cl_mem), &location_buf1[i*input_config[batch_size]+k]);
						checkError(status, "Failed to set argument %d of kernel locationRd", argi - 1);
						printf("ERROR: load location from location_buf%d, not defined.\n", layer_config[j][locationRd_src]);
					}
				}

				////**********************************************************
				//// Set knl_conv arguments
				////**********************************************************
				argi = 0;
				output_num = layer_config[j][conv_x]*layer_config[j][conv_y]; 
				conv_loop_cnt = layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE;

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &output_num);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &conv_loop_cnt);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &control);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_char), &precision_config[j][frac_din]);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_char), &precision_config[j][frac_dout]);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_char), &layer_config[j][relu_on]);
				checkError(status, "Failed to set argument %d of kernel conv", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_w]);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_h]);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &layer_config[j][weight_n]);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &weight_dim4_div_lane);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &weight_dim1x2);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &weight_dim2x3);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint),  &weight_dim1x2x3);
				checkError(status, "Failed to set argument %d of kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &layer_config[j][conv_x]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &layer_config[j][conv_y]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_mem), &weights_buf[i*LAYER_NUM+j]);
				checkError(status, "Failed to set argument %d kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_mem), &bias_buf[i*LAYER_NUM+j]);
				checkError(status, "Failed to set argument %d kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_mem), &frac_bias_buf[i*LAYER_NUM+j]);
				checkError(status, "Failed to set argument %d kernel weightsload", argi - 1);

				status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_mem), &frac_weight_buf[i*LAYER_NUM+j]);
				checkError(status, "Failed to set argument %d kernel weightsload", argi - 1);
		
				////**********************************************************
				//// Set knl_pool arguments
				////**********************************************************
				if (layer_config[j][pool_on]) {
					argi = 0;
					input_num = layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM; 
					line_size = layer_config[j][conv_x];
					odd_flag_x = layer_config[j][conv_x]%2;
					odd_flag_y = layer_config[j][conv_y]%2;
								
					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &layer_config[j][conv_x]);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
		
					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &layer_config[j][conv_y]);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &input_num);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &line_size);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &layer_config[j][pool_size]);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &layer_config[j][pool_stride]);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &odd_flag_x);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
					
					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_ushort), &odd_flag_y);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);

					status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &control);
					checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
				}
				////**********************************************************
				//// Set knl_unpooling arguments
				////**********************************************************
				if (layer_config[j][unpooling_on]){
					argi = 0;

					unpooling_outnum = layer_config[j][unpooling_x]*layer_config[j][unpooling_y]*layer_config[j][weight_m]/LANE_NUM;
					unpooling_dim3 = layer_config[j][weight_m]/LANE_NUM;

					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &unpooling_outnum);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);          

					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &layer_config[j][unpooling_x]);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);          

					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &layer_config[j][unpooling_y]);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);                

					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &unpooling_dim3);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);   

					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &layer_config[j][unpooling_odd_flag_y]);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);
					
					status = clSetKernelArg(knl_unpooling[i], argi++, sizeof(cl_uint), &control);
					checkError(status, "Failed to set argument %d of kernel unpooling", argi - 1);											
				}
				
				////**********************************************************
				//// Set knl_memWr arguments
				////**********************************************************
				argi = 0;
				////// Set the sizes of work_item, which are based on the size of the data that should be storaged
				if((control&0x03)==0x01){
					memWr_dim1 = layer_config[j][pool_x];
					memWr_dim2 = layer_config[j][pool_y];
					memWr_dim3 = layer_config[j][weight_m];
				}
				else if((control&0x02)==0x02){
					memWr_dim1 = layer_config[j][unpooling_x];
					memWr_dim2 = layer_config[j][unpooling_y];
					memWr_dim3 = layer_config[j][weight_m];
				}
				else{
					memWr_dim1 = layer_config[j][conv_x];
					memWr_dim2 = layer_config[j][conv_y];
					memWr_dim3 = layer_config[j][weight_m];
				}

				batch_size_in_dim  = 1;
				batch_indx_dim1 = 0;
				batch_indx_dim2 = 0;
				out_dim1xbatch = memWr_dim1*batch_size_in_dim;
				out_dim1x2xbatch = memWr_dim1*memWr_dim2*batch_size_in_dim*batch_size_in_dim;
				padding_offset_top = (layer_config[j][weight_m]-layer_config_original[j][weight_m])/2;
				padding_offset_bottom = (layer_config[j][weight_m]-layer_config_original[j][weight_m])-padding_offset_top;

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim1);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim2);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &memWr_dim3); 
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_ushort), &out_dim1xbatch);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uint), &out_dim1x2xbatch);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &batch_indx_dim1);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &batch_indx_dim2);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uint), &control);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &padding_offset_top);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &padding_offset_bottom);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				
				// Select the kernel output mem object source
				if(layer_config[j][memwr_dst] ==0){
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				else{
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				if(layer_config[j][locationWr_dst] == 1){
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &location_buf1[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				else if(layer_config[j][locationWr_dst] == 2){
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &location_buf2[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				else if(layer_config[j][locationWr_dst] == 3){
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &location_buf3[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				else if(layer_config[j][locationWr_dst] == 4){
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &location_buf4[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				} 
				else{
					status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &location_buf1[i*input_config[batch_size]+k]);
					checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
				}
				// Excutes Kernel
				status = clEnqueueTask(que_memRd[i], knl_memRd[i], 0, NULL, &memRd_event[i]);
				checkError(status, "Failed to launch kernel memRd");
				if(status != CL_SUCCESS) cleanupall();
				//// Kernel locationRd
				if ((control & 0x03) == 0x02) {
					status = clEnqueueTask(que_locationRd[i], knl_locationRd[i], 0, NULL, &locationRd_event[i]);
					checkError(status, "Failed to launch kernel locationRd");
					if(status != CL_SUCCESS) cleanupall();
				}
				//// Kernel conv
				status = clEnqueueTask(que_conv[i], knl_conv[i], 0, NULL, &conv_event[i]);
				checkError(status, "Failed to launch kernel conv");
				if(status != CL_SUCCESS) cleanupall();
				//// Kernel pool
				if (layer_config[j][pool_on]) {
					status = clEnqueueTask(que_pool[i], knl_pool[i], 0, NULL, &pool_event[i]);
					checkError(status, "Failed to launch kernel pool");
					if(status != CL_SUCCESS) cleanupall();
				}
				//// Kernel unpooling
				if (layer_config[j][unpooling_on]) {
					status = clEnqueueTask(que_unpooling[i], knl_unpooling[i], 0, NULL, &unpooling_event[i]);
					checkError(status, "Failed to launch kernel unpooling");
					if(status != CL_SUCCESS) cleanupall();
				}
				//// kernel memWr
				knl_memWr_global_size[0] = memWr_dim1;
				knl_memWr_global_size[1] = memWr_dim2;
				knl_memWr_global_size[2] = memWr_dim3;
				knl_memWr_local_size[0] = 1;
				knl_memWr_local_size[1] = 1;
				knl_memWr_local_size[2] = LANE_NUM;

				status = clEnqueueNDRangeKernel( que_memWr[i], knl_memWr[i], 3, NULL, knl_memWr_global_size, knl_memWr_local_size, 0, NULL, &memWr_event[i]);
				checkError(status, "Failed to launch kernel memWr");
				if(status != CL_SUCCESS) cleanupall();

				// Wait for all kernel to finish
				status = clWaitForEvents(num_devices, memWr_event);
				checkError(status, "Failed to finish memWr event");
				if(status != CL_SUCCESS) cleanupall();

				// Profile mode, get excution time for each kernel
				memRd_time[j] += getKernelStartEndTime(memRd_event[i]);
				if ((control & 0x03) == 0x02)
				locationRd_time[j]  += getKernelStartEndTime(locationRd_event[i]);
				conv_time[j] += getKernelStartEndTime(conv_event[i]);
				if (layer_config[j][pool_on])
				pool_time[j] += getKernelStartEndTime(pool_event[i]);
				if(layer_config[j][unpooling_on])
				unpooling_time[j] += getKernelStartEndTime(unpooling_event[i]);
				memWr_time[j] += getKernelStartEndTime(memWr_event[i]);
				// Must release event object to avoid performance
				// degeneration !!!
				clReleaseEvent(memRd_event[i]);
				checkError(status, "Failed to release memRd event object");
				if(status != CL_SUCCESS) cleanupall();
				if ((control & 0x03) == 0x02) {
					clReleaseEvent(locationRd_event[i]);
					checkError(status, "Failed to release locationRd event object");
					if(status != CL_SUCCESS) cleanupall();
				}
				clReleaseEvent(conv_event[i]);
				checkError(status, "Failed to release conv event object");
				if(status != CL_SUCCESS) cleanupall();
				if (layer_config[j][pool_on]) {
					status = clReleaseEvent(pool_event[i]);
					checkError(status, "Failed to release pool event object");
					if(status != CL_SUCCESS) cleanupall();
				}
				if (layer_config[j][unpooling_on]) {
					status = clReleaseEvent(unpooling_event[i]);
					checkError(status, "Failed to release uppoolling event object");
					if(status != CL_SUCCESS) cleanupall();
				}
				clReleaseEvent(memWr_event[i]);
				checkError(status, "Failed to release memWR event object");
				if(status != CL_SUCCESS) cleanupall();
			} // end of batch             
		} // end of layers
		// Recorde the end time
		t.stop();
		time = t.get_time_s();        
		printf("Done! Inference time is %fs  \n", time);
		readDataBack();       
		verifyResult();
  	}// end of board iteration
	double max_kernel_time_out = 0.0f;
	double batch_double = double(input_config[batch_size]);
	// Release resource
	cv::Mat prediction_map(TEST_DATA_HEIGHT, TEST_DATA_WIDTH, CV_8UC1);
	for(int i=0;i<TEST_DATA_HEIGHT;i++){
		for(int j=0;j<TEST_DATA_WIDTH;j++){
			prediction_map.at<uchar>(i,j) = ind[i][j];
		}
	}
	return prediction_map;       
}

int Classifier::prepare() {
	// Load Image data, CNN net weights and golden_results
	ifstream bin_file_r;
	unsigned file_size;
	unsigned weight_size;
	unsigned output_size;
	unsigned output__reorder_size;
	unsigned int weight_ptr = 0; // original weight and bias offset for each layer
	unsigned int frac_ptr = 0; // original weight and bias offset for each layer
	unsigned char conv_win_size_dim1, conv_win_size_dim2;
	unsigned padding_offset[LAYER_NUM];
	// Parameter initialization and safty check
	for (unsigned ll = 0; ll < LAYER_NUM; ll++) {
		// First, backup the original layer configurations
		for (unsigned ii = 0; ii < NUM_CONFIG_ITEM; ii++) {
			layer_config_original[ll][ii] = layer_config[ll][ii];
		}
		// Second, perform padding on dim4, when it is not divisible by LANE_NUM
		if (layer_config[ll][weight_m] % LANE_NUM != 0) {
			printf("\nWarnning: layer-%d requires padding zero-value feature  maps for give param LANE_NUM=%d\n", ll + 1, LANE_NUM);
			layer_config[ll][weight_m] = ceil((float)layer_config[ll][weight_m] / LANE_NUM) * LANE_NUM;
			layer_config[ll][bias_size] = layer_config[ll][weight_m];
			layer_config[ll][frac_bias_size] = layer_config[ll][weight_m];
			layer_config[ll][frac_weight_size] = layer_config[ll][weight_m];
			printf("original num of feature maps is %d, new value is %d\n", layer_config_original[ll][weight_m], layer_config[ll][weight_m]);
			// padding of weight on dim4 is needed
			padding_offset[ll] = layer_config[ll][weight_m] - layer_config_original[ll][weight_m];
			// check if evenly padding on two sides is possible
			if (((layer_config[ll][weight_m] / LANE_NUM) % 2 != 0) & (layer_config[ll][conv_split] == 1)) {
				printf("Error: could not perform padding for split mode, weight_m/LANE_NUM must be divisible by 2 !!!\n\n");
				return 1;
			} 
			else { // padding zeros evenly on two sides of dim4
				padding_offset[ll] = padding_offset[ll] / 2;
				printf("padding_offset=%d (layer=%d)\n\n", padding_offset[ll], ll + 1);
			}
		} 
		else {
			padding_offset[ll] = 0;
		}
		// Check parameters
		if (ll == 0) { // check parameters for layer-1
			if (input_config[image_w] != layer_config_original[ll][data_w] ||
				input_config[image_h] != layer_config_original[ll][data_h] ||
				input_config[image_n] != layer_config_original[ll][data_n]) {
				printf("Error: incorrect layer configuration img-w,h,n,for layer-%d !!!\n", ll + 1);
				return 1;
			}
			if ((layer_config_original[ll][weight_n] != input_config[image_n])) {
				printf("\nError: incorrect layer configuration weight-n for layer-%d !!!\n", ll + 1);
				return 1;
			}
		} 
		else { // other layers
			// Currently weight_n must be divisible by VEC_SIZE (for first layer, padding is performed when weight_n is not divisible by VEC_SIZE)
			if ((layer_config[ll][weight_n] % VEC_SIZE) != 0) {
				printf("\nError: incorrect setting of parameter VEC_SIZE !!!\n");
				return 1;
			}
			if ((layer_config_original[ll][data_n] != layer_config_original[ll - 1][conv_z])) {
				printf("\nError: incorrect setting of convolution input/output size for layer-%d!!!\n", ll + 1);
				return 1;
			}
		}
		if ((layer_config_original[ll][conv_x] != (layer_config_original[ll][data_w] - layer_config_original[ll][weight_w]
																							+ 2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride] + 1) ||
				(layer_config_original[ll][conv_y] != (layer_config_original[ll][data_h] - layer_config_original[ll][weight_h]
																							+ 2*layer_config_original[ll][conv_padding])/layer_config_original[ll][conv_stride] + 1) ||
				(layer_config_original[ll][conv_z] != layer_config_original[ll][weight_m])) {
			printf("\nError: incorrect setting of convolution output size or filter params for layer-%d!!!\n", ll + 1);
			return 1;
		}
		if (layer_config_original[ll][pool_on]){
			if(	(layer_config_original[ll][pool_x] != ceil((float)(layer_config_original[ll][conv_x] - layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]) + 1) ||
					(layer_config_original[ll][pool_y] != ceil((float)(layer_config_original[ll][conv_y] - layer_config_original[ll][pool_size])/layer_config_original[ll][pool_stride]) + 1) ||
					(layer_config_original[ll][pool_z] != layer_config_original[ll][conv_z])) {
				printf("\nError: incorrect setting of pooling input/output size for layer-%d!!!\n", ll + 1);
				return 1;
			}
		}
		if (layer_config[ll][conv_x] == 1) { // when only one group for FC layer
			conv_win_size_dim1 = layer_config[ll][weight_w];
		} 
		else {
			conv_win_size_dim1 = layer_config[ll][weight_w] + (CONV_GP_SIZE_X - 1) * layer_config[ll][conv_stride];
		}
		conv_win_size_dim2 = layer_config[ll][weight_h];
		// check win_buffer size
		if (conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE > WIN_BUF_SIZE) {
			printf("Error: required win_buffer size is %d, configured size is %d\n",
				conv_win_size_dim1*conv_win_size_dim2*layer_config[ll][weight_n]/VEC_SIZE, WIN_BUF_SIZE);
			return 1;
		}
		// check weight_buffer size
		if (layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE > WEIGHT_BUF_SIZE) {
			printf("Error: required weight_buffer size is %d, configured size is %d \n",
				layer_config[ll][weight_w]*layer_config[ll][weight_h]*layer_config[ll][weight_n]/VEC_SIZE, WEIGHT_BUF_SIZE);
			return 1;
		}
  	}    
	weights = (DTYPE *)alignedMalloc(sizeof(DTYPE) * WEIGHTS_FILE_SIZE, DMA_ALIGNMENT);
	frac 	= (DTYPE *)alignedMalloc(sizeof(DTYPE) * FRAC_FILE_SIZE, DMA_ALIGNMENT);
	image   = (DTYPE *)alignedMalloc(sizeof(DTYPE)*IMAGE_FILE_SIZE, DMA_ALIGNMENT);//mean!!!                               
	// input data buffers
	// padding the input RGB image with extra number of zeros channels, so that data_n/weight_n is divisible by VEC_SIZE
	layer_config[0][weight_n] = ceil((float)layer_config[0][weight_n] / VEC_SIZE) * VEC_SIZE;
	layer_config[0][data_n]   = layer_config[0][weight_n];
	data_init = (DTYPE *)alignedMalloc(sizeof(DTYPE) * layer_config[0][data_w] * layer_config[0][data_h] * layer_config[0][data_n], DMA_ALIGNMENT);
	memset(data_init, 0, sizeof(DTYPE) * layer_config[0][data_w] * layer_config[0][data_h] * layer_config[0][data_n]);				// fill non-RGB dims with 0

	// final results
	if (LAYER_NUM >= CONV_NUM) {// For last conv and all fc layers, all batch results are read back
		output_size = output_config[output_w] * output_config[output_h] * layer_config[LAYER_NUM-1][weight_m] * input_config[batch_size];
		output__reorder_size = output_config[output_w] * output_config[output_h] * output_config[output_n] * input_config[batch_size];
	}			
	else 														// For other conv layers, only one item of result are read back
		output_size 		 = output_config[output_w] * output_config[output_h] * layer_config[LAYER_NUM-1][weight_m];
		output__reorder_size = output_config[output_w] * output_config[output_h] * output_config[output_n];
		output 		      	 = (DTYPE *)alignedMalloc(sizeof(DTYPE) * output_size, DMA_ALIGNMENT);               // vectorized results
		output_reorder 		 = (DTYPE *)alignedMalloc(sizeof(DTYPE) * output__reorder_size, DMA_ALIGNMENT);      // reordered results for verifying
	if (image == NULL || weights == NULL || data_init == NULL || output == NULL || frac == NULL || output_reorder == NULL) {
		printf("Not enough memory !!!");
		alignedFree(weights);
		alignedFree(frac);
		alignedFree(data_init);
		alignedFree(image);//mean!!!
		alignedFree(output);
		alignedFree(output_reorder);
		return 1;
	}
	for (int j = 0; j < LAYER_NUM; j++) {
		weight_size 		= (layer_config[j][weight_w] * layer_config[j][weight_h] * layer_config[j][weight_n] * layer_config[j][weight_m]);
		weight_conv[j] 		= (DTYPE *)alignedMalloc(sizeof(DTYPE) * weight_size, DMA_ALIGNMENT);
		bias_conv[j] 		= (DTYPE *)alignedMalloc(sizeof(DTYPE) * layer_config[j][bias_size], DMA_ALIGNMENT);
		frac_bias_conv[j] 	= (DTYPE *)alignedMalloc(sizeof(DTYPE) * layer_config[j][frac_bias_size], DMA_ALIGNMENT);
		frac_weight_conv[j] = (DTYPE *)alignedMalloc(sizeof(DTYPE) * layer_config[j][frac_weight_size], DMA_ALIGNMENT);
		
		memset(weight_conv[j], 0, sizeof(DTYPE) * weight_size); 														// reset all value (include padding value) to zero
		memset(bias_conv[j], 0, sizeof(DTYPE) * layer_config[j][bias_size]); 								// reset all value (include padding value) to zero
		memset(frac_bias_conv[j], 0, sizeof(DTYPE) * layer_config[j][frac_bias_size]);      // reset all value (include padding value) to zero             
		memset(frac_weight_conv[j], 0, sizeof(DTYPE) * layer_config[j][frac_weight_size]);	// reset all value (include padding value) to zero
						
		if (weight_conv[j] == NULL || bias_conv[j] == NULL || frac_weight_conv[j] == NULL || frac_bias_conv[j] == NULL) {
			printf("Not enough memory !!!");
			for (int i = 0; i <= j; i++) {
				alignedFree(weight_conv[i]);
				alignedFree(bias_conv[i]);
				alignedFree(frac_bias_conv[i]);
				alignedFree(frac_weight_conv[i]);
			}               
			return 1;
		}
	}
	// Weights
	bin_file_r.open(weights_file, ios::in | ios::binary);
	if (bin_file_r.is_open()) {
		// Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);
		bin_file_r.read((char *)weights, sizeof(DTYPE) * WEIGHTS_FILE_SIZE);
		printf("\n%d total weights read \n", file_size / ((int)sizeof(DTYPE)));
		if (WEIGHTS_FILE_SIZE != (file_size / (sizeof(DTYPE))))
			printf("Warning: weight file size does not match user configuration !!!\n");
		bin_file_r.close();
	} else
		printf("Weights file does not exits !!!\n");

	// frac_w frac_b_ch
	bin_file_r.open(frac_file, ios::in | ios::binary);
	if (bin_file_r.is_open()) {
		// Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		printf("file_size: %d\n",file_size);
		bin_file_r.seekg(0, bin_file_r.beg);
		bin_file_r.read((char *)frac, sizeof(DTYPE) * FRAC_FILE_SIZE);
		printf("\n%d total frac read \n", file_size / ((int)sizeof(DTYPE)));
		if (FRAC_FILE_SIZE != (file_size / (sizeof(DTYPE))))
			printf("Warning: frac file size does not match user configuration !!!\n");
		bin_file_r.close();
	} else
		printf("Frac file does not exits !!!\n");

	mean_data = (float *) malloc(sizeof(float)*MEAN_DATA_WIDTH*MEAN_DATA_HEIGHT*MEAN_DATA_CHANNEl);
	bin_file_r.open(mean_file, ios::in | ios::binary);
	if (bin_file_r.is_open()) {
		// Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);
		bin_file_r.read((char *)mean_data, sizeof(float) * MEAN_DATA_WIDTH*MEAN_DATA_HEIGHT*MEAN_DATA_CHANNEl);
		bin_file_r.close();
	} else
		printf("mean file does not exits !!!\n");
	// Layer-1
	reorderWeights(weights, weight_conv[0], layer_config[0][weight_w], layer_config[0][weight_h], layer_config[0][weight_n], layer_config[0][weight_m],
									layer_config_original[0][weight_n], layer_config_original[0][weight_m], weight_ptr, padding_offset[0], VEC_SIZE, LANE_NUM);
	weight_ptr 	+= layer_config[0][weight_w] * layer_config[0][weight_h] * layer_config_original[0][weight_n] * layer_config_original[0][weight_m];

	reorderBias(weights, bias_conv[0], weight_ptr, padding_offset[0], layer_config[0][bias_size], layer_config_original[0][bias_size], LANE_NUM);
	weight_ptr 	+= layer_config_original[0][bias_size];
	// frac 
	reorderBias(frac, frac_weight_conv[0], frac_ptr, padding_offset[0], layer_config[0][frac_weight_size], layer_config_original[0][frac_weight_size], LANE_NUM);
	frac_ptr 	+= layer_config_original[0][frac_weight_size];

	reorderBias(frac, frac_bias_conv[0], frac_ptr, padding_offset[0], layer_config[0][frac_bias_size], layer_config_original[0][frac_bias_size], LANE_NUM);
	frac_ptr 	+= layer_config_original[0][frac_bias_size];

	// Other layers
	for (unsigned j = 1; j < LAYER_NUM; j++) {
		if (weight_ptr + layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config_original[j][weight_n]*layer_config_original[j][weight_m]>WEIGHTS_FILE_SIZE){
			printf("Error：exceed weight file size !!!\n");
			printf("weight_ptr=%d,j=%d\n",weight_ptr,j);
			return 1;
		}
		if ((frac_ptr + layer_config_original[j][frac_bias_size] + layer_config_original[j][frac_weight_size]) > FRAC_FILE_SIZE) {
			printf("Error：exceed frac file size !!!\n");
			printf("frac_ptr=%d,j=%d\n",frac_ptr,j);
			return 1;
		}
		reorderWeights(weights, weight_conv[j], layer_config[j][weight_w], layer_config[j][weight_h], layer_config[j][weight_n], layer_config[j][weight_m], layer_config_original[j][weight_n], layer_config_original[j][weight_m], weight_ptr, padding_offset[j], VEC_SIZE, LANE_NUM);
		weight_ptr 	+= layer_config[j][weight_w] * layer_config[j][weight_h] * layer_config_original[j][weight_n] * layer_config_original[j][weight_m];
		reorderBias(weights, bias_conv[j], weight_ptr, padding_offset[j], layer_config[j][bias_size], layer_config_original[j][bias_size], LANE_NUM);
		weight_ptr 	+= layer_config_original[j][bias_size];
		reorderBias(frac, frac_weight_conv[j], frac_ptr, padding_offset[j], layer_config[j][frac_weight_size], layer_config_original[j][frac_weight_size], LANE_NUM);
		frac_ptr 	+= layer_config_original[j][frac_weight_size];
		reorderBias(frac, frac_bias_conv[j], frac_ptr, padding_offset[j], layer_config[j][frac_bias_size], layer_config_original[j][frac_bias_size], LANE_NUM);
		frac_ptr    += layer_config_original[j][frac_bias_size];
	}
	return 0;    
}

void Classifier::loadImageToBuffer(const cv::Mat& img) {
	cl_int status;
	imshow("original_image", img);
	//waitKey(0);
	Mat img1;
	resize(img,img1,Size(MEAN_DATA_WIDTH,MEAN_DATA_HEIGHT));
	img1.convertTo(img1,CV_32FC3);
	Mat mean_mat(MEAN_DATA_WIDTH, MEAN_DATA_HEIGHT, CV_32FC3, mean_data);
	img1 = img1 - mean_mat;  
	// resize to the input size of the first layer
	Mat img2;
	resize(img1,img2,Size(layer_config_original[0][data_w],layer_config_original[0][data_h]));
	// convert to 8-bit fixed-point
	img2.convertTo(img2,CV_8SC3);
	// reorder channel sequence from RGB to GBR
	DTYPE * data_ptr = (DTYPE*)img2.data;
	unsigned int w,h,c;
	unsigned int k=0;
	for(h=0;h<layer_config_original[0][data_h];h++){
		for(w=0;w<layer_config_original[0][data_w];w++){
			for (c=0;c<layer_config_original[0][data_n];c++){
				image[c*layer_config_original[0][data_w]*layer_config_original[0][data_h]+h*layer_config_original[0][data_w]+w]=data_ptr[k];
				k++;
			}
		}
	}

	for(unsigned n = 0; n<layer_config[0][data_n]/VEC_SIZE; n++){
		for(unsigned i = 0; i<layer_config[0][data_h]; i++){
			for(unsigned j = 0; j<layer_config[0][data_w]; j++){
				for(unsigned k = 0; k<VEC_SIZE; k++){
					if((n*VEC_SIZE+k)<layer_config_original[0][data_n]){ //  when layer_config[0][data_n] > layer_config_original[0][data_n], only copy valid pixels
						data_init[n*VEC_SIZE*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w]*VEC_SIZE + j*VEC_SIZE + k]
							= (DTYPE) image[(n*VEC_SIZE+k)*layer_config[0][data_h]*layer_config[0][data_w] + i*layer_config[0][data_w] + j];
					}
				}
			}
		}
	}

	for (unsigned i = 0; i < num_devices; ++i) {
		// Create data buffers for each batch item
		for (unsigned j = 0; j < input_config[batch_size]; ++j) {
			// Load image data into buffers
			status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i * input_config[batch_size] + j], CL_TRUE, 0, (layer_config[0][data_w] * layer_config[0][data_h] * layer_config[0][data_n]) * sizeof(DTYPE), data_init, 0, NULL, NULL);
			checkError(status, "Failed to transfer input image");
		}
	}
} 

void Classifier::reorderWeights(DTYPE *weights, DTYPE *weight_buf, unsigned dim1,
									unsigned dim2, unsigned dim3, unsigned dim4,
									unsigned dim3_original, unsigned dim4_original,
									unsigned offset, unsigned padding_offset, unsigned vecSize,
									unsigned laneNum) {

	DTYPE *copy_with_padding;
	// First, copy the data into new buffer and padding in dim3/dim4 with zeros
	// if needed
	copy_with_padding = (DTYPE *)malloc(sizeof(DTYPE) * dim1 * dim2 * dim3 * dim4);
	if (copy_with_padding == NULL) {
		printf("Error: not enough memory when padding weight!!!");
		free(copy_with_padding);
	}
	memset(copy_with_padding, 0, sizeof(DTYPE) * dim1 * dim2 * dim3 * dim4);

	for (unsigned m = 0; m < dim4_original; m++) {
		for (unsigned n = 0; n < dim3_original; n++) {
			for (unsigned i = 0; i < dim2; i++) {
				for (unsigned j = 0; j < dim1; j++) {
					copy_with_padding[(padding_offset * dim1 * dim2 * dim3) + m * dim1 * dim2 * dim3 + n * dim1 * dim2 + i * dim1 + j] = (DTYPE)
							weights[offset + m * dim1 * dim2 * dim3_original + n * dim1 * dim2 + i * dim1 + j];
				}
			}
		}
	}
	// Second, perform vectorization in dim3 by VEC_SIZE and at the same time,
	// perform vectorization in dim4 by a factor of LANE_NUM
	for (unsigned m = 0; m < (dim4 / laneNum); m++) {
		for (unsigned n = 0; n < (dim3 / vecSize); n++) {
			for (unsigned i = 0; i < dim2; i++) {
				for (unsigned j = 0; j < dim1; j++) {
					for (unsigned ll = 0; ll < laneNum; ll++) {
						for (unsigned k = 0; k < vecSize; k++) {
							weight_buf[m * dim1 * dim2 * dim3 * laneNum + n * dim1 * dim2 * vecSize * laneNum + i * dim1 * vecSize * laneNum + j * vecSize * laneNum + ll * vecSize + k] =
								(DTYPE)copy_with_padding[(m * laneNum + ll) * dim3 * dim2 * dim1 + (n * vecSize + k) * dim1 * dim2 + i * dim1 + j];
						}
					}
				}
			}
		}
	}
	// release resource
	free(copy_with_padding);
}

void Classifier::reorderBias(DTYPE *dataIn, DTYPE *bias, unsigned offset,
                 unsigned padding_offset, unsigned dim4, unsigned dim4_original,
                 unsigned laneNum) {

	DTYPE *copy_with_padding;
	// first copy the data into new buffer with zero paddings
	copy_with_padding = (DTYPE *)malloc(sizeof(DTYPE) * dim4);
	if (copy_with_padding == NULL) {
		printf("Not enough memory when reordering bias!!!");
		free(copy_with_padding);
	}
	memset(copy_with_padding, 0, sizeof(DTYPE) * dim4);
	// padding evenly on two sides of weight_m
	memcpy(copy_with_padding + padding_offset, dataIn + offset, sizeof(DTYPE) * dim4_original);
	// second, perform vectorization by factor of LANE_NUM
	for (unsigned m = 0; m < (dim4 / laneNum); m++) {
		for (unsigned ll = 0; ll < laneNum; ll++) {
			bias[m * laneNum + ll] = (DTYPE)copy_with_padding[m * laneNum + ll];
		}
	}
	// release resource
	free(copy_with_padding);
}

void Classifier::readDataBack() {
	unsigned int read_buf_size;
	cl_int status;
	scoped_array<cl_event> finish_event(num_devices);
	// Read back the results from the device to verify the output
	// Note：only device0 is used here
	if (num_devices != 1)
		printf("Warnning: only the result from device0 will be verified!!!\n\n");
	// Select whith item you would like to compare with the golden ref
	// Item num start from 0
	unsigned batch_item_num = 0;
	if (batch_item_num > (input_config[batch_size] - 1)) {
		printf("Error: wrong configuration，can't verify the item since it is layer than batch size !!!\n\n");
	}

	if (LAYER_NUM < CONV_NUM) { // verify conv results
		read_buf_size = output_config[output_w] * output_config[output_h] * layer_config[LAYER_NUM-1][weight_m];
	} 
	else // verify the last conv and all fc results
		read_buf_size = output_config[output_w] * output_config[output_h] * layer_config[LAYER_NUM-1][weight_m] * input_config[batch_size];

	// For the last conv layer and all fc layers, read result from data buffer or output buffer
	if (layer_config[LAYER_NUM - 1][memwr_dst] == 1) {
		printf("\nCopyed one result from NO.%d output buffers.\n", batch_item_num);
		std::chrono::steady_clock::time_point tt1 = std::chrono::steady_clock::now();
		status = clEnqueueReadBuffer(que_memWr[0], output_buf[batch_item_num], CL_FALSE, 0, sizeof(DTYPE) * read_buf_size, output, 0, NULL, &finish_event[0]);
		checkError(status, "Failed to set transfer output data");
		std::chrono::steady_clock::time_point tt2 = std::chrono::steady_clock::now();
		double t_clEnqueueReadBuffer= std::chrono::duration_cast<std::chrono::duration<double> >(tt2 - tt1).count();
		cout << " clEnqueueReadBuffer time  =" << t_clEnqueueReadBuffer*1000 << "ms" << endl;
		if(status != CL_SUCCESS) cleanupall();
	} 
	else {
		printf("\nCopyed one results from NO.%d data buffers.\n", batch_item_num);
		std::chrono::steady_clock::time_point tt3 = std::chrono::steady_clock::now();
		status = clEnqueueReadBuffer(que_memWr[0], data_buf[batch_item_num], CL_FALSE, 0, sizeof(DTYPE) * read_buf_size, output, 0, NULL, &finish_event[0]);      
		checkError(status, "Failed to set transfer output data");
		std::chrono::steady_clock::time_point tt4 = std::chrono::steady_clock::now();
		double tt_clEnqueueReadBuffer= std::chrono::duration_cast<std::chrono::duration<double> >(tt4 - tt3).count();
		cout << " clEnqueueReadBuffer time  =" << tt_clEnqueueReadBuffer*1000 << "ms" << endl;
		if(status != CL_SUCCESS) cleanupall();
	}
	// Wait for reads to finish
	clWaitForEvents(num_devices, &finish_event[0]);
	clReleaseEvent(finish_event[0]);
	checkError(status, "Failed to release finish event object");
	if(status != CL_SUCCESS) cleanupall();

	unsigned padding_top = (layer_config[LAYER_NUM-1][weight_m] - layer_config_original[LAYER_NUM-1][weight_m])/2;
	unsigned padding_bottom = (layer_config[LAYER_NUM-1][weight_m] - layer_config_original[LAYER_NUM-1][weight_m]) - padding_top;
	reorderOutput(output, output_reorder, output_config[output_w], output_config[output_h], layer_config[LAYER_NUM-1][weight_m], padding_top, padding_bottom);  
}
                              

void Classifier::reorderOutput(DTYPE *output, DTYPE *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3, unsigned padding_top, unsigned padding_bottom) {
	unsigned dim3_item;
	for (unsigned k = 0; k < dim3/VEC_SIZE; k++) {
		for (unsigned i = 0; i < dim2; i++) {
			for (unsigned j = 0; j < dim1; j++) {
				for (unsigned vv = 0; vv < VEC_SIZE; vv++){
					dim3_item = k * VEC_SIZE + vv;
					if(dim3_item < dim3 - padding_top -padding_bottom) {
						output_reorder[dim3_item * dim2 * dim1 + i * dim1 + j] =
										output[k * dim2 * dim1 * VEC_SIZE + i * dim1 * VEC_SIZE + j * VEC_SIZE + vv]; 
					}
				} 
			}
		}
	}
}

void Classifier::verifyResult(){
	//static double test_output[TEST_DATA_CHANNEL] = {0.0};
	static float Matrix_Rev[TEST_DATA_HEIGHT * TEST_DATA_WIDTH][TEST_DATA_CHANNEL];
	for (unsigned i = 0; i < TEST_DATA_CHANNEL; ++i) {
		for (unsigned j = 0; j < TEST_DATA_HEIGHT * TEST_DATA_WIDTH; ++j) {
			Matrix_Rev[j][i] = (float)output_reorder[i * TEST_DATA_HEIGHT * TEST_DATA_WIDTH + j];   
		}
	}
	//	test_output_txt(output_reorder);
	float max[TEST_DATA_HEIGHT][TEST_DATA_WIDTH] = {0.0};
	for (unsigned i = 0; i < TEST_DATA_HEIGHT; ++i) {
		for (unsigned j = 0; j < TEST_DATA_WIDTH; ++j) {
			//max[i][j]=0;
			for (unsigned k = 0; k < TEST_DATA_CHANNEL; ++k) {
				if(max[i][j] < Matrix_Rev[i * TEST_DATA_WIDTH + j][k]){
					max[i][j] = Matrix_Rev[i * TEST_DATA_WIDTH + j][k];
					ind[i][j] = k;
				}
			}
		}
	}
	int label_colours[21][3] = {{0, 0, 0}, {128,0,0}, {0, 128, 0},
															{128, 128, 0},  {0, 0, 128}, {128, 0, 128},
															{0, 128, 128}, {128, 128, 128}, {64, 0, 0},
															{192, 0, 0}, {64, 128, 0}, {192, 128, 0},
															{64, 0, 128}, {192, 0, 128}, {64, 128, 128},
															{192, 128, 128}, {0, 64, 0}, {128, 64, 0},
															{0, 192, 0}, {128, 192, 0}, {0, 64, 128}
															};
	static int r[TEST_DATA_HEIGHT][TEST_DATA_WIDTH] = {0};
	static int g[TEST_DATA_HEIGHT][TEST_DATA_WIDTH] = {0};
	static int b[TEST_DATA_HEIGHT][TEST_DATA_WIDTH] = {0};
	for (unsigned i = 0; i < TEST_DATA_HEIGHT; ++i) {
		for (unsigned j = 0; j < TEST_DATA_WIDTH; ++j) {
			for (unsigned k = 0; k < 21; k++) {         //!!!
				if (ind[i][j] == k) {
					r[i][j] = label_colours[k][0];
					g[i][j] = label_colours[k][1];
					b[i][j] = label_colours[k][2];
				}
			}
		}
	}
	static float rgb[TEST_DATA_HEIGHT][TEST_DATA_WIDTH][3] = {0.0};
	memset(rgb, 0.0, sizeof(rgb));
	for (int i = 0; i < TEST_DATA_HEIGHT; ++i) {
		for (int j = 0; j < TEST_DATA_WIDTH; ++j) {
			rgb[i][j][0] = (float)r[i][j];
			rgb[i][j][1] = (float)g[i][j];
			rgb[i][j][2] = (float)b[i][j];
		}
	}
	Mat result_dst;
	result_dst.create(224,224,CV_8UC3);
	for (unsigned i = 0; i < TEST_DATA_HEIGHT; ++i) {
		for (unsigned j = 0; j < TEST_DATA_WIDTH; ++j) {
			result_dst.at<Vec3b>(i, j)[0] = rgb[i][j][2];
			result_dst.at<Vec3b>(i, j)[1] = rgb[i][j][1];
			result_dst.at<Vec3b>(i, j)[2] = rgb[i][j][0];
		}
	}
	imshow("result_img", result_dst);
	cvMoveWindow("result_img", 800,0);
	//waitKey(0);
	waitKey(30);

}

void Classifier::cleanupall() {
	// Release the opencl runtime resource allocated
	for (unsigned i = 0; i < num_devices; ++i) {
		if (knl_memRd && knl_memRd[i]) {
			clReleaseKernel(knl_memRd[i]);
		}
		if (knl_locationRd && knl_locationRd[i]) {
			clReleaseKernel(knl_locationRd[i]);
		}        
		if (knl_conv && knl_conv[i]) {
			clReleaseKernel(knl_conv[i]);
		}
		if (knl_pool && knl_pool[i]) {
			clReleaseKernel(knl_pool[i]);
		}
		if (knl_unpooling && knl_unpooling[i]) {
			clReleaseKernel(knl_unpooling[i]);
		}
		if (knl_memWr && knl_memWr[i]) {
			clReleaseKernel(knl_memWr[i]);
		}
		if (que_memRd && que_memRd[i]) {
			clReleaseCommandQueue(que_memRd[i]);
		}
		if (que_locationRd && que_locationRd[i]) {
			clReleaseCommandQueue(que_locationRd[i]);
		}
		if (que_conv && que_conv[i]) {
			clReleaseCommandQueue(que_conv[i]);
		}
		if (que_pool && que_pool[i]) {
			clReleaseCommandQueue(que_pool[i]);
		}
		if (que_unpooling && que_unpooling[i]) {
			clReleaseCommandQueue(que_unpooling[i]);
		}
		if (que_memWr && que_memWr[i]) {
			clReleaseCommandQueue(que_memWr[i]);
		}
		if (data_buf && data_buf[i]) {
			clReleaseMemObject(data_buf[i]);
		}
		if (output_buf && output_buf[i]) {
			clReleaseMemObject(output_buf[i]);
		}
		if (weights_buf && weights_buf[i]) {
			clReleaseMemObject(weights_buf[i]);
		}
		if (bias_buf && bias_buf[i]) {
			clReleaseMemObject(bias_buf[i]);
		}
		if (frac_bias_buf && frac_bias_buf[i]) {
			clReleaseMemObject(frac_bias_buf[i]);
		}
		if (frac_weight_buf && frac_weight_buf[i]) {
			clReleaseMemObject(frac_weight_buf[i]);
		}
		if (location_buf1 && location_buf1[i]) {
			clReleaseMemObject(location_buf1[i]);
		}
		if (location_buf2 && location_buf2[i]) {
			clReleaseMemObject(location_buf2[i]);
		}
		if (location_buf3 && location_buf3[i]) {
			clReleaseMemObject(location_buf3[i]);
		}
		if (location_buf4 && location_buf4[i]) {
			clReleaseMemObject(location_buf4[i]);
		}
	}
	if (program) {
		clReleaseProgram(program);
	}
	if (context) {
		clReleaseContext(context);
	}
	alignedFree(weights);
	alignedFree(frac);
	alignedFree(data_init);
	alignedFree(mean_data);
	for (int j = 0; j < LAYER_NUM; j++) {
		alignedFree(weight_conv[j]);
		alignedFree(bias_conv[j]);
		alignedFree(frac_bias_conv[j]);
		alignedFree(frac_weight_conv[j]);
	} 
	alignedFree(output);
	alignedFree(output_reorder); 
}



