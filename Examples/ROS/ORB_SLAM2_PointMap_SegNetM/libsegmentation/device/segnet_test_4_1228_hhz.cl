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

#define USE_ROM
#include "segnet_hw_param666_W.cl"
#include "rtl_lib.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable

// Define the precision of the data-path
typedef char	DPTYPE;
typedef int		MACTYPE;

// Vectorized data type
typedef struct {
   	DPTYPE data[VEC_SIZE];	
} lane_data;

// Combined vec-data type from multiple lane
typedef struct {
   	lane_data lane[LANE_NUM];
} channel_vec;			

// Combined scalar data type from multiple lane
typedef struct {
   	DPTYPE lane[LANE_NUM];
} channel_scal;			

channel lane_data		data_ch[WEIGHT_W]			__attribute__((depth(VEC_SIZE)));		
channel channel_scal	data_ch_location_c		__attribute__((depth(8)));
channel channel_scal	data_ch_location_p		__attribute__((depth(8)));
channel channel_scal	pool_ch								__attribute__((depth(8)));
channel channel_scal	bypass_ch							__attribute__((depth(8)));
channel channel_scal	pooling_ch						__attribute__((depth(8)));
channel channel_scal	unpooling_ch					__attribute__((depth(8)));	
channel channel_scal	unpooling_ch_in_c			__attribute__((depth(8)));
channel channel_scal	unpooling_ch_in_p			__attribute__((depth(8)));
channel channel_scal	pooling_ch_location		__attribute__((depth(8)));

// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data weights)
{
	MACTYPE output = MASK_MULT & CZERO;
	#pragma unroll
	for(int i=0; i<VEC_SIZE/4; i++){
		output = output + (MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4],
														input.data[i*4+1], weights.data[i*4+1],
														input.data[i*4+2], weights.data[i*4+2],
														input.data[i*4+3], weights.data[i*4+3]));
	}
	return output;
}

DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
	DPTYPE max_value;	
	if(a_in >= b_in)
		max_value = a_in;
	else
		max_value = b_in;	
	return max_value;
}

// Fetch Data from Global Memory
__kernel
__attribute__((max_global_work_dim(0)))		//设置内核不使用任何全局，本地或组ID，内核是单个工作项内核。
void memRead(
	// Params Ports
	uint	data_dim1,				//data_w
	uint	data_dim2,				//data_h
	uint	data_dim1xdim2,			//data_w*data_h
	uchar  	weight_dim1,			//weight_w
	uchar  	weight_dim2,			//weight_h
	uint 	weight_dim3,			//weight_n
	uint 	weight_dim4_div_lane, 	// avoid generating divider,weight_m/lane_num
	uint 	weight_dim2x3,			//weight_h*weight_n
	uint   	weight_dim1x2x3,		//weight_w*weight_h*weight_n
	uint	conv_x,					//conv_x
	uchar  	stride,					//步长
	uchar  	padding,				//data块需要补零的个数
	uchar  	split,					//weight的channel维度上分的块数
	uint	group_num_x,			//data_x上能划过窗的数目
	uint	group_num_y,			//data_y上分块的个数
	uchar	group_rem_x,
	uint  	group_rem_size_x_rd,	//最后一个窗实际有多少个feature
	uint  	group_rem_size_x,		//最后一个窗实际有多少个feature
	uint  	group_rem_size_xyz,		//最后一个窗实际有多少个feature
	uint   	group_rem_size_xyz_out,	//最后一个窗实际输出的次数
	uint  	win_size_x,				//data切好的小块在x维度上有几个feature（已补零）
	uint  	win_size_y,				//data切好的小块在y维度上有几个feature，weight_h
	uint   	win_size_xyz, 			//data切好的小块总共有多少个feature 
	uint   	win_size_xyz_out, 		//data切好的小块输出的次数 			
	// Data Ports
	__global lane_data     *restrict bottom
)

{
	// Input Data, Weights and Bias, bn_beta, bn_gamma, frac_b, frac_w
	lane_data     data_vec;
	lane_data     data_vec_out[WEIGHT_W];	
	channel_vec   data_ch_vec;
	channel_scal  bias_ch_in;	
	channel_scal  frac_b_ch_in;
	channel_scal  frac_w_ch_in;
	
	// virtual loop counters
	uint 	data_offset = 0; 	// assuming the 1st layer is not in split	
	uint	gp_num_x, gp_num_y, out_idx_z;
	uint	gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
	uint	output_idx_dim2;	
	uint	output_idx_dim3;
	uint	win_itm_x, win_itm_y, win_itm_z;	
	uint  	gp_item_idx_x;
	uint	feature_idx_dim1, feature_idx_dim2, feature_idx_dim3;
	uint   	item_loop_bound, item_loop_bound_rdx, item_loop_bound_wtx, item_loop_wtx;
	uchar  	flag; 				// ping-pong flag
	uchar	load_flag;
	// Ping-pong buffer
	__local lane_data    win_buffer[WIN_BUF_SIZE][2]	__attribute__((bankwidth(VEC_SIZE), numbanks(2)));	// working sequence 0->1->0->1 ...
	gp_num_x_winbuf = 0; 		// there is only one group for FC mode when batch=1;x维度上分块的计数

	// reset global group virtual loop counters
	gp_num_x = 0;				// x维度方向上窗的计数
	gp_num_y = 0;		
	out_idx_z = 0;				// LANE_NUM维度上的计数
	
	Group:for(unsigned int out_idx_xyz=0; out_idx_xyz<=(weight_dim4_div_lane*group_num_y*group_num_x); out_idx_xyz++) {	//以conv中含有多少个group作为计数
		flag = out_idx_xyz & 0x01; 	// ping-pong flag
		load_flag = 1;				
		// reset output loop counters
		output_idx_dim2 = 0;
		output_idx_dim3 = 0;		//输出数据在channel维度上的计数(/VEC_SIZE)
		// reset in-group item counters 
		gp_item_idx_x = 0;			//在一个块内x维度上窗的计数，或者理解为卷积结果x维度上在一个CONV_GP_SIZE_X内feature的计数（第几个窗）
		// reset input winbuffer loop counters
		win_itm_x = 0;				//输入在一个窗(item工作项)内x维度的计数
		win_itm_y = 0;
		win_itm_z = 0;	
		if(gp_num_x==group_num_x-1)	{
			item_loop_bound = group_rem_size_xyz>=win_size_xyz_out?(group_rem_size_xyz/VEC_SIZE):(win_size_xyz_out/VEC_SIZE);					//循环上限运算，算data块将有多少数被拿来输出
			item_loop_bound_rdx = group_rem_size_x_rd;			
			item_loop_bound_wtx = win_size_x;
			item_loop_wtx = CONV_GP_SIZE_X;
		}											
		else if((gp_num_x==group_num_x-1) && (out_idx_xyz>0)) {
			item_loop_bound = win_size_xyz/VEC_SIZE;	//循环上限运算，算data块将有多少数被拿来输出
			item_loop_bound_rdx = win_size_x;			
			item_loop_bound_wtx = group_rem_size_x_rd;
			item_loop_wtx = group_rem_x;
		}
		else {
			item_loop_bound = win_size_xyz/VEC_SIZE;
			item_loop_bound_rdx = win_size_x;			
			item_loop_bound_wtx = win_size_x;
			item_loop_wtx = CONV_GP_SIZE_X;
		}
		#pragma ivdep array(win_buffer)		
		Item:for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {		//以一个CONV_GP_SIZE（group）中有多少个feature作为计数
			feature_idx_dim1 = win_itm_x+gp_num_x*CONV_GP_SIZE_X*stride;
			feature_idx_dim2 = win_itm_y+gp_num_y*CONV_GP_SIZE_Y*stride;
			feature_idx_dim3 = win_itm_z;
			// Winbuffer loading operations
			if(out_idx_xyz<weight_dim4_div_lane*group_num_y*group_num_x) {
				if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)) {		
					data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];		
				}
				else {		// for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
					#pragma unroll
					for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
						data_vec.data[vv] = CZERO;										
					}
				}
				win_buffer[win_itm_z*win_size_y*item_loop_bound_rdx + win_itm_y*item_loop_bound_rdx + win_itm_x][(~flag)&0x01] = data_vec;	//1->0->1->0->……
				if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1) && (win_itm_x==item_loop_bound_rdx-1))
					win_itm_z = 0;
				else if((win_itm_y==win_size_y-1) && (win_itm_x==item_loop_bound_rdx-1))
					win_itm_z++;
				if((win_itm_y==win_size_y-1) && (win_itm_x==item_loop_bound_rdx-1))
					win_itm_y = 0;
				else if(win_itm_x==item_loop_bound_rdx-1)
					win_itm_y++;
				if(win_itm_x==item_loop_bound_rdx-1)
					win_itm_x = 0;
				else
					win_itm_x++;								
			}					
			// In this version, grouping is only performed in row (x) direction
			if((gp_num_x_winbuf*CONV_GP_SIZE_X+gp_item_idx_x<conv_x) && (out_idx_xyz>0)) {        
				if(load_flag) {		
					#pragma unroll
					data_vec_out[0] = win_buffer[output_idx_dim3*win_size_y*item_loop_bound_wtx + output_idx_dim2*item_loop_bound_wtx + (gp_item_idx_x*stride)][flag];
					data_vec_out[1] = win_buffer[output_idx_dim3*win_size_y*item_loop_bound_wtx + output_idx_dim2*item_loop_bound_wtx + (1+gp_item_idx_x*stride)][flag];
					data_vec_out[2] = win_buffer[output_idx_dim3*win_size_y*item_loop_bound_wtx + output_idx_dim2*item_loop_bound_wtx + (2+gp_item_idx_x*stride)][flag];
					write_channel_intel(data_ch[0], data_vec_out[0]);
					write_channel_intel(data_ch[1], data_vec_out[1]);
					write_channel_intel(data_ch[2], data_vec_out[2]);
				}
				// used as output loop counters
				if((gp_item_idx_x==item_loop_wtx-1) && (output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1)) {
					load_flag = 0;
				}
				if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1)) {
					output_idx_dim3 = 0;
					gp_item_idx_x++;
				}
				else if(output_idx_dim2==weight_dim2-1)
					output_idx_dim3++;
			
				if(output_idx_dim2==weight_dim2-1)
					output_idx_dim2 = 0;
				else
					output_idx_dim2++;			
			}
		}
		// used as virtual group loop counters for winbuf loading operations
		if((out_idx_z_winbuf==weight_dim4_div_lane-1) && (gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			out_idx_z_winbuf = 0;
		else if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			out_idx_z_winbuf++;	
		if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
			gp_num_y_winbuf = 0;
		else if(gp_num_x_winbuf==group_num_x-1)
			gp_num_y_winbuf++;	
		if(gp_num_x_winbuf==group_num_x-1)
			gp_num_x_winbuf = 0;
		else if(out_idx_xyz>0)
			gp_num_x_winbuf++;
		// used as virtual group loop counters
		if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			out_idx_z = 0;
		else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			out_idx_z++;
		if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
			gp_num_y = 0;
		else if(gp_num_x==group_num_x-1)
			gp_num_y++;
		if(gp_num_x==group_num_x-1)
			gp_num_x = 0;
		else
			gp_num_x++;
	}	
}

// Fetch Location from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))		//设置内核不使用任何全局，本地或组ID，内核是单个工作项内核。
void locationRd(
			// Params Ports
			ushort	location_dim1,			//location_w
			uint		location_group_num,	//location_h*location_n/LANE_NUM
			// Data Ports
			__global	channel_scal  *restrict bottom_location
				)
{	
	channel_scal	location_scal_in;
	channel_scal	location_scal_out;
	unsigned char	gp_num_idx=0;
	unsigned char	flag;       			// ping-pong flag	
	__local channel_scal	location_buffer[LOC_MAX_BUF_SIZE][2]	__attribute__((bankwidth(LANE_NUM), numbanks(2)));	// working sequence 0->1->0->1 ...  
	// Initialize the winbuf with the data in the first iteration of the group of data,取第一块数据
	for(unsigned short  location_idx_x=0; location_idx_x<location_dim1; location_idx_x++) {
		//Load location from global memory to channel
		location_scal_in = bottom_location[location_idx_x];	
		location_buffer[location_idx_x][0] = location_scal_in;
	}
	gp_num_idx = 1;							//第一个group已经读进来了
	Group:for(unsigned int gp_idx_xyz=gp_num_idx; gp_idx_xyz<=location_group_num; gp_idx_xyz++) {
		flag = gp_idx_xyz & 0x01;
		Item:for(unsigned int location_idx_x=0; location_idx_x<location_dim1; location_idx_x++) {
			//Load location from global memory to buffer
			if(gp_idx_xyz<location_group_num) {
				location_scal_in = bottom_location[gp_idx_xyz*location_dim1+location_idx_x];			
				location_buffer[location_idx_x][flag] = location_scal_in;                 
      	}      				
    		//Load location from buffer to channel
			location_scal_out = location_buffer[location_idx_x][(~flag)&0x01];		
			write_channel_intel(data_ch_location_c, location_scal_out);						
		}
	}
}

__kernel
__attribute__((max_global_work_dim(0)))	
void coreConv(
	// Params Ports
	uint  	output_num,				//conv_x*y
	uint  	conv_loop_cnt,			//h*n/V_S
	uint  	control, 				//[0]-> bypass [1]-> pooling [2]-> unpooling [3]->middle
	char  	frac_din,
	char  	frac_dout,
	char  	relu_on,
	uchar  	weight_dim1,			//weight_w
	uchar  	weight_dim2,			//weight_h
	uint	weight_dim3,			//weight_n
	uint	weight_dim4_div_lane, 	// avoid generating divider,weight_m/lane_num
	uint 	weight_dim1x2,			//weight_w*weight_h
	uint 	weight_dim2x3,			//weight_h*weight_n			
	uint   	weight_dim1x2x3,		//weight_w*weight_h*weight_n	
	uint	conv_x,					//conv_x
	uint	conv_y,					//conv_y						
	// Data Ports
	__global 	channel_vec   *restrict weights,
	__global 	channel_scal  *restrict bias,
	__global 	channel_scal  *restrict frac_b,
	__global 	channel_scal  *restrict frac_w			
	)
{
	lane_data		mac_data_in[WEIGHT_W];
	channel_vec		mac_data[WEIGHT_W];
	channel_vec 	weight_vec;
 	channel_vec 	mac_weight[WEIGHT_W];
	channel_vec  	weight_ch_vec[WEIGHT_W];	
	channel_scal  	bias_ch_in;		 
	channel_scal 	bias_ch_out;
	channel_scal 	frac_b_ch_out;
	channel_scal 	frac_w_ch_out;
	channel_scal 	conv_ch_in;
	channel_scal 	frac_b_ch_in;
	channel_scal  	frac_w_ch_in;	

	//uchar loadflag; //!!!!!!
	DPTYPE		conv_quantization[LANE_NUM];
	DPTYPE  	conv_final[LANE_NUM];
	DPTYPE		relu_result[LANE_NUM];
	MACTYPE 	lane_accum_0[LANE_NUM], lane_accum_1[LANE_NUM], lane_accum_2[LANE_NUM];
	MACTYPE 	accum_piped_0[LANE_NUM][PIPE_DEPTH];
	MACTYPE 	accum_piped_1[LANE_NUM][PIPE_DEPTH];
	MACTYPE 	accum_piped_2[LANE_NUM][PIPE_DEPTH];
	MACTYPE 	conv_out_mid[LANE_NUM], conv_out_mid_0[LANE_NUM], conv_out_mid_1[LANE_NUM], conv_out_mid_2[LANE_NUM];	
	MACTYPE 	conv_out[LANE_NUM];
	MACTYPE 	conv_sign_exten[LANE_NUM];
	MACTYPE 	conv_with_rnd_bit[LANE_NUM];
	MACTYPE 	conv_sum_bias[LANE_NUM];

	// Local memory
	__local channel_vec		weights_buffer_0[MAX_WEIGHT_LAYER_BUF_SIZE]		__attribute__((bankwidth(VEC_SIZE*LANE_NUM)));
	__local channel_vec		weights_buffer_1[MAX_WEIGHT_LAYER_BUF_SIZE]		__attribute__((bankwidth(VEC_SIZE*LANE_NUM)));
	__local channel_vec		weights_buffer_2[MAX_WEIGHT_LAYER_BUF_SIZE]		__attribute__((bankwidth(VEC_SIZE*LANE_NUM)));
	__local channel_scal  	bias_buffer[BIAS_BUF_SIZE/LANE_NUM]				__attribute__((bankwidth(LANE_NUM)));
  	__local channel_scal  	frac_b_buffer[BIAS_BUF_SIZE/LANE_NUM]			__attribute__((bankwidth(LANE_NUM)));
	__local channel_scal  	frac_w_buffer[BIAS_BUF_SIZE/LANE_NUM]			__attribute__((bankwidth(LANE_NUM)));

	// Load weights, bias, frac_b, frac_w from global memory into local buffer
  	for(unsigned int wei_itm_m=0; wei_itm_m<weight_dim4_div_lane; wei_itm_m++) {
		bias_buffer[wei_itm_m]		= bias[wei_itm_m];
		frac_b_buffer[wei_itm_m] 	= frac_b[wei_itm_m];
		frac_w_buffer[wei_itm_m] 	= frac_w[wei_itm_m];
		for(unsigned int wei_itm_z=0; wei_itm_z<weight_dim3/VEC_SIZE; wei_itm_z++) {	
			for(unsigned int  wei_itm_y=0; wei_itm_y<weight_dim2; wei_itm_y++) {
				for(unsigned int  wei_itm_x=0; wei_itm_x<weight_dim1; wei_itm_x++) {
					uint	weight_idx = wei_itm_m*weight_dim1x2x3/VEC_SIZE + wei_itm_z*weight_dim1x2 + wei_itm_y*weight_dim1 + wei_itm_x;
					for(unsigned char ll=0; ll<LANE_NUM; ll++) {
						weight_vec.lane[ll] = weights[weight_idx].lane[ll];							
					}
					if(wei_itm_x==0) {
						weights_buffer_0[wei_itm_z*weight_dim2 + wei_itm_y] = weight_vec;
					}
					else if(wei_itm_x==1) {
						weights_buffer_1[wei_itm_z*weight_dim2 + wei_itm_y] = weight_vec;
					}
					else {
						weights_buffer_2[wei_itm_z*weight_dim2 + wei_itm_y] = weight_vec;
					}					
				}
			}
		} 
		// each iteration generates one output
		for(unsigned int k=0; k<output_num; k++) {
			#pragma unroll
			for(unsigned char ll=0; ll<LANE_NUM; ll++) {		
				conv_out[ll] 	= CZERO_32;
				conv_out_mid_0[ll] = CZERO_32;
				conv_out_mid_1[ll] = CZERO_32;
				conv_out_mid_2[ll] = CZERO_32;					
				#pragma unroll
				for(unsigned char p=0; p<PIPE_DEPTH; p++) {
					accum_piped_0[ll][p] = CZERO_32;
					accum_piped_1[ll][p] = CZERO_32;
					accum_piped_2[ll][p] = CZERO_32;
				}				
			}
			bias_ch_out		= bias_buffer[wei_itm_m];		//一次读LANE_NUM个bias	
			frac_b_ch_out	= frac_b_buffer[wei_itm_m];		//一次读LANE_NUM个frac_b
			frac_w_ch_out 	= frac_w_buffer[wei_itm_m];		//一次读LANE_NUM个frac_w		
			#pragma ivdep array(weights_buffer_0)
			#pragma ivdep array(weights_buffer_1)
			#pragma ivdep array(weights_buffer_2)
			for(unsigned int j=0; j<conv_loop_cnt; j++)	{	//读data和weight相应的数乘加，channel维度取/VEC_SIZE计数，LANE_NUM维度取一组(LANE_NUM)计数
				weight_ch_vec[0]	= weights_buffer_0[j];
				weight_ch_vec[1] 	= weights_buffer_1[j];
				weight_ch_vec[2] 	= weights_buffer_2[j];
				mac_data_in[0]	 	= read_channel_intel(data_ch[0]);
				mac_data_in[1]	 	= read_channel_intel(data_ch[1]);
				mac_data_in[2]	 	= read_channel_intel(data_ch[2]);
				mac_weight[0]		= weight_ch_vec[0];				
				mac_weight[1]		= weight_ch_vec[1];
				mac_weight[2]		= weight_ch_vec[2];
				#pragma unroll 
				for(unsigned char ll=0; ll<LANE_NUM; ll++) {
					mac_data[0].lane[ll] 	= mac_data_in[0];
					mac_data[1].lane[ll] 	= mac_data_in[1];
					mac_data[2].lane[ll] 	= mac_data_in[2];
					lane_accum_0[ll] 		= accum_piped_0[ll][PIPE_DEPTH-1] + mac(mac_data[0].lane[ll], mac_weight[0].lane[ll]);	
					lane_accum_1[ll] 		= accum_piped_1[ll][PIPE_DEPTH-1] + mac(mac_data[1].lane[ll], mac_weight[1].lane[ll]);	
					lane_accum_2[ll] 		= accum_piped_2[ll][PIPE_DEPTH-1] + mac(mac_data[2].lane[ll], mac_weight[2].lane[ll]);	
					#pragma unroll
					for(unsigned char p=PIPE_DEPTH-1; p>0; p--) {
						accum_piped_0[ll][p] = accum_piped_0[ll][p-1];
						accum_piped_1[ll][p] = accum_piped_1[ll][p-1];
						accum_piped_2[ll][p] = accum_piped_2[ll][p-1];
					}	
					accum_piped_0[ll][0] = lane_accum_0[ll];
					accum_piped_1[ll][0] = lane_accum_1[ll];
					accum_piped_2[ll][0] = lane_accum_2[ll];
				}

			}// end of conv loop
			#pragma unroll
			for(unsigned char ll=0; ll<LANE_NUM; ll++) {
				#pragma unroll
				for(unsigned char i=0; i<PIPE_DEPTH; i++) {
					conv_out_mid_0[ll] = conv_out_mid_0[ll]+accum_piped_0[ll][i];
					conv_out_mid_1[ll] = conv_out_mid_1[ll]+accum_piped_1[ll][i];
					conv_out_mid_2[ll] = conv_out_mid_2[ll]+accum_piped_2[ll][i];
				}
				conv_out_mid[ll] = conv_out_mid_0[ll] + conv_out_mid_1[ll] + conv_out_mid_2[ll];
				conv_out[ll] = conv_out[ll]+conv_out_mid[ll];
				//quantization
				if(conv_out[ll]>=0)
					conv_sign_exten[ll] = 0x00;
				else
					conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w_ch_out.lane[ll]+frac_din-frac_dout-1));

				conv_with_rnd_bit[ll] = (conv_sign_exten[ll] | (conv_out[ll]>>(frac_w_ch_out.lane[ll]+frac_din-frac_dout-1))) + 0x01;

				if(conv_with_rnd_bit[ll]>=256)
					conv_sum_bias[ll] = MASK9B & 0xFF;
				else if(conv_with_rnd_bit[ll]<-256)
					conv_sum_bias[ll] = MASK9B & 0x100;
				else
					conv_sum_bias[ll] = (MASK9B & conv_with_rnd_bit[ll])+(bias_ch_out.lane[ll]>>(frac_b_ch_out.lane[ll]-frac_dout-1))+0x01;

				conv_quantization[ll] = MASK8B & (conv_sum_bias[ll]>>0x01);
				conv_final[ll] = conv_quantization[ll];				
				// Relu operation
				if(relu_on==1){
					if((conv_final[ll]&MASKSIGN)==MASKSIGN)
						relu_result[ll] = 0;
					else
						relu_result[ll] = conv_final[ll];
				}
				else
					relu_result[ll] = conv_final[ll]; 
							
				conv_ch_in.lane[ll] = relu_result[ll];	
			}				
			// write convoluation results
			if((control&0x01)==0x01){
				// to pooling kernel
				write_channel_intel(pool_ch, conv_ch_in);
			}
			else if((control&0x03)==0x02){
				// to unpooling kernel
				write_channel_intel(unpooling_ch_in_c, conv_ch_in);
			}
			else{
				// by-pass
				write_channel_intel(bypass_ch, conv_ch_in);
			}
		}// end of output loop
  	}	
}

// Pooling
__kernel
__attribute__((task))
void maxPool(
	// Params Ports
	ushort	input_x,
	ushort	input_y,
	uint	input_num,			//layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][weight_m]/LANE_NUM
	ushort	line_size,			// line_size should be no larger than LOC_MAX_BUF_SIZE
	ushort	pool_size,			// by now, only pooling size = 2
	ushort	pool_stride,
	ushort	pool_odd_flag_x,	// odd_flag_x = conv_x && 0x01, if the conv_x is an odd number, odd_flag_y == 1
	ushort	pool_odd_flag_y,	// odd_flag_y = conv_y && 0x01, if the conv_y is an odd number, odd_flag_y == 1
	uint  	control				//[0]-> bypass [1]-> pooling [2]-> unpooling [3]->middle
	)	
{
	ushort	line_buf_ptr;
	ushort	col_pool_cnt;
	ushort	row_pool_cnt;
	ushort	row_cnt;
	uchar	ch_location;
	channel_scal	conv_ch_out;
	channel_scal	conv_ch_out_location;
	channel_scal	pool_final;
	channel_scal	pool_final_location;
	channel_scal	line_buf_0[LOC_MAX_BUF_SIZE];
	channel_scal	line_buf_0_location[LOC_MAX_BUF_SIZE];
	channel_scal	row_pool_reg;
	channel_scal	row_pool_reg_location;
	channel_scal	col_pool_reg;
	channel_scal	col_pool_reg_location;
	channel_scal	pool_reg[POOL_MAX_SIZE];
	channel_scal	pool_reg_location[POOL_MAX_SIZE];
	
	// Each iteration consumes one output from convolution kernel and then Pooling is performed in column and row directions
	line_buf_ptr = 0;
	row_pool_cnt = 0;
	col_pool_cnt = 0;
	row_cnt		 = 0;
	
	for(unsigned int k=0; k<input_num; k++){
		conv_ch_out = read_channel_intel(pool_ch);
		// Two line buffer to form the 3x3 pooling window
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			ch_location = (((row_pool_cnt & 0x01)<<1) + (line_buf_ptr & 0x01)) & 0x03;
			conv_ch_out_location.lane[ll] = ch_location;
      		row_pool_reg.lane[ll] = line_buf_0[line_buf_ptr].lane[ll];
      		row_pool_reg_location.lane[ll] = line_buf_0_location[line_buf_ptr].lane[ll];
    		pool_reg[0].lane[ll] = pool_max(row_pool_reg.lane[ll], conv_ch_out.lane[ll]);
      		pool_reg_location[0].lane[ll] = row_pool_reg.lane[ll]>=conv_ch_out.lane[ll]?(row_pool_reg_location.lane[ll]):(conv_ch_out_location.lane[ll]); 
      		col_pool_reg.lane[ll] = pool_reg[1].lane[ll];
      		col_pool_reg_location.lane[ll] = pool_reg_location[1].lane[ll];
      		if((line_buf_ptr==(input_x-1)) && (pool_odd_flag_x == 1) && (row_cnt==(input_y-1)) && (pool_odd_flag_y == 1)) {
				pool_final.lane[ll] = conv_ch_out.lane[ll];
				pool_final_location.lane[ll] = conv_ch_out_location.lane[ll];
      		}
      		else if((row_cnt==(input_y-1)) && (pool_odd_flag_y == 1)) {
				pool_final.lane[ll] = pool_max(col_pool_reg.lane[ll], conv_ch_out.lane[ll]);
				pool_final_location.lane[ll] = col_pool_reg.lane[ll]>=pool_reg[0].lane[ll]?(col_pool_reg_location.lane[ll]):(conv_ch_out_location.lane[ll]);
      		}
      		else if((line_buf_ptr==(input_x-1)) && (pool_odd_flag_x == 1)) {
				pool_final.lane[ll] = pool_reg[0].lane[ll];
				pool_final_location.lane[ll] = pool_reg_location[0].lane[ll];
      		}
      		else {
				pool_final.lane[ll] = pool_max(col_pool_reg.lane[ll], pool_reg[0].lane[ll]);
				if(col_pool_reg.lane[ll] > pool_reg[0].lane[ll])
					pool_final_location.lane[ll] = col_pool_reg_location.lane[ll];
				else if(col_pool_reg.lane[ll] == pool_reg[0].lane[ll])
					pool_final_location.lane[ll] = col_pool_reg_location.lane[ll]>pool_reg_location[0].lane[ll]?(pool_reg_location[0].lane[ll]):(col_pool_reg_location.lane[ll]);
				else
					pool_final_location.lane[ll] = col_pool_reg.lane[ll]>=pool_reg[0].lane[ll]?(col_pool_reg_location.lane[ll]):(pool_reg_location[0].lane[ll]);
      		}
      		line_buf_0[line_buf_ptr].lane[ll] = conv_ch_out.lane[ll];
      		line_buf_0_location[line_buf_ptr].lane[ll] = conv_ch_out_location.lane[ll];
      		#pragma unroll
      		for(unsigned char p=POOL_MAX_SIZE-1; p>0; p--){
        		pool_reg[p].lane[ll]=pool_reg[p-1].lane[ll];
        		pool_reg_location[p].lane[ll] = pool_reg_location[p-1].lane[ll];
    		}
		}
		// Generates pooling pipeline register wr/rd pointer
		if((control&0x03)==0x03) {
			if(row_pool_cnt==(pool_size-1)) {
				if(((line_buf_ptr==input_x-1) && (pool_odd_flag_x == 1)) || ((col_pool_cnt==pool_size-1) && (line_buf_ptr>=pool_size-1))) {
					write_channel_intel(unpooling_ch_in_p, pool_final);
					write_channel_intel(data_ch_location_p, pool_final_location);				
				}
				if(col_pool_cnt==pool_size-1)
					col_pool_cnt = (pool_size-pool_stride);
				else
					col_pool_cnt = col_pool_cnt + 1;				
			}
			else
				col_pool_cnt = 0;				
		}
		else {
			if(row_pool_cnt==(pool_size-1)) {
				if(((line_buf_ptr==input_x-1) && (pool_odd_flag_x == 1)) || ((col_pool_cnt==pool_size-1) && (line_buf_ptr>=pool_size-1))) {
					write_channel_intel(pooling_ch, pool_final);
					write_channel_intel(pooling_ch_location, pool_final_location);					
				}
				if(col_pool_cnt==pool_size-1)
					col_pool_cnt = (pool_size-pool_stride);
				else
					col_pool_cnt = col_pool_cnt + 1;				
			}
			else
				col_pool_cnt = 0;				
		}
		// Generates line buffer wr/rd pointer
		if(line_buf_ptr==(line_size-1)){
			line_buf_ptr = 0;			
			// Row counters for recognize frames
			if(row_cnt == (input_y-1)){ // assuming row_num = line_size, i.e. rectangular frame
				row_cnt = 0;
			}
			else
				row_cnt = row_cnt + 1;
			// Pooling window slide counter for rows
			if(row_cnt == 0)
				row_pool_cnt = 0;
			else if(row_pool_cnt==(pool_size-1)){
				if((row_cnt==(input_y-1)) && (pool_odd_flag_y == 1))	
					row_pool_cnt = (pool_size-1);
				else	
					row_pool_cnt = (pool_size-pool_stride);
		    	}				
			else
				row_pool_cnt = row_pool_cnt + 1;
		}
		else{
			line_buf_ptr = line_buf_ptr + 1;
		}
	}
}

// Unpooling
__kernel
__attribute__((task))
void unpooling(
	// Params Ports、
	uint	output_num,
	uint	unpooling_dim1,			// unpooling_w
	uint	unpooling_dim2,			// unpooling_h
	uint	unpooling_dim3,			// unpooling_n/LANE_NUM
	uint	unpooling_oddflag_y,	// if unpooling_oddflag_y = 1, the last line should be unpooling by 1*2
	uint  	control					// [0]-> bypass [1]-> pooling [2]-> unpooling [3]->middle
	)
{
  	// Params
	unsigned int	output_cnt;
  	// Register
	channel_scal	unpooling_in;
	channel_scal	unpooling_in_location;
	channel_scal	unpooling_out;
	channel_scal	unpooling_line_buf_0[2][LINE_MAX_SIZE];	// ping-pong buffer, 2 lines 
	channel_scal	unpooling_line_buf_1[2][LINE_MAX_SIZE];				
	// Local memory
	// NONE	
	// Counters
	unsigned int	row_total_cnt;
	unsigned int	row_cnt_unpooling;
	unsigned int	inline_buf_cnt;
	unsigned int	outline_buf_cnt;
	// Flags
	unsigned int	linebuf_flag;							// ping-pong buffer flag
	unsigned int	in_cnt_mod2;
	unsigned int	out_cnt_mod2;
	unsigned int	in_reg_flag[2];
	unsigned int	out_line_flag;

	output_cnt			= output_num + (2 + unpooling_dim3 * unpooling_oddflag_y) * unpooling_dim1;
	row_total_cnt 		= 0;
	row_cnt_unpooling	= 0;
	inline_buf_cnt		= 0;
	outline_buf_cnt		= 0;
	linebuf_flag		= 0;
	out_line_flag		= 0;

	for(unsigned int unpooling_k = 0; unpooling_k < output_cnt; unpooling_k++) {
		linebuf_flag	= linebuf_flag & 0x01;
		in_cnt_mod2		= inline_buf_cnt & 0x01;
		out_cnt_mod2	= outline_buf_cnt & 0x01;
		// Data read
		if((unpooling_k < output_num + unpooling_dim3 * unpooling_dim1 * unpooling_oddflag_y) && (in_cnt_mod2 == 0) && (out_cnt_mod2 == 0)) {
			if(((control&0x03)==0x03)) {
				unpooling_in 			= read_channel_intel(unpooling_ch_in_p);
				unpooling_in_location 	= read_channel_intel(data_ch_location_p);	
			}
			else {
				unpooling_in 			= read_channel_intel(unpooling_ch_in_c);
				unpooling_in_location 	= read_channel_intel(data_ch_location_c);		
			}	
		}
		in_reg_flag[0] = in_cnt_mod2 & 0x03;
		in_reg_flag[1] = (0x02 + in_cnt_mod2) & 0x03;
		if(unpooling_k < output_num + 2 * unpooling_dim1 * unpooling_oddflag_y) {
			channel_scal	unpooling_in_reg[2];
			#pragma unroll
			for(unsigned char ll = 0; ll < LANE_NUM; ll++) {
				// Data to unpooling_in_reg[0]
				if(unpooling_in_location.lane[ll] 	== in_reg_flag[0]) {
					unpooling_in_reg[0].lane[ll] 	= unpooling_in.lane[ll];
				}
				else {
					unpooling_in_reg[0].lane[ll] 	= CZERO;
				}			
				// Data to unpooling_in_reg[1]
				if(unpooling_in_location.lane[ll] 	== in_reg_flag[1]) {
					unpooling_in_reg[1].lane[ll] 	= unpooling_in.lane[ll];
				}
				else {
					unpooling_in_reg[1].lane[ll] 	= CZERO;
				}		
				// Data put into line_buf
				unpooling_line_buf_0[linebuf_flag][inline_buf_cnt].lane[ll] = unpooling_in_reg[0].lane[ll];
				unpooling_line_buf_1[linebuf_flag][inline_buf_cnt].lane[ll] = unpooling_in_reg[1].lane[ll];					
			}
		}
		// Data output from line_buf
		if((row_total_cnt >=2) && (row_cnt_unpooling < unpooling_dim2)) {
			if(out_line_flag) {
				unpooling_out = unpooling_line_buf_1[(~linebuf_flag)&0x01][outline_buf_cnt];
			}
			else {
				unpooling_out = unpooling_line_buf_0[(~linebuf_flag)&0x01][outline_buf_cnt];
			}
			write_channel_intel(unpooling_ch, unpooling_out);
		}		
		// Cnt and Flag update
		if(outline_buf_cnt == unpooling_dim1 - 1) {
			row_total_cnt++;
			if((row_cnt_unpooling >= unpooling_dim2 - 1) && (row_cnt_unpooling & 0x01)) {
				row_cnt_unpooling = 0;
			}
			else if(row_total_cnt>2) {
				row_cnt_unpooling++;
			}
			if(inline_buf_cnt == unpooling_dim1 - 1) {
				out_line_flag = 0;
				linebuf_flag++;
			}
			else {
				out_line_flag = 1;
			}
			outline_buf_cnt = 0;
			if(inline_buf_cnt == unpooling_dim1 - 1) {
				inline_buf_cnt = 0;	
			}
			else if(out_cnt_mod2 == 1) {
				inline_buf_cnt++;
			}
		}
		else {
			if(out_cnt_mod2 == 1) {
				inline_buf_cnt++;
			}
			outline_buf_cnt++;
		}					
	}
}

// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
	// Params Ports
	ushort	out_dim1,			// memWr_dim1
	ushort	out_dim2,			// memWr_dim2
	ushort	out_dim3,			// memWr_dim3
	ushort	out_dim1xbatch, 	// memWr_dim1
	uint	out_dim1x2xbatch, 	// memWr_dim1*memWr_dim2
	uchar	batch_indx_dim1,	// ==0
	uchar	batch_indx_dim2,	// ==0
	uint	control, 			// [0]-> bypass [1]-> pooling [2]-> unpooling [3]->middle
	uchar	padd_offset_top,	// ((layer_config[j][weight_m]-layer_config_original[j][weight_m])/2
	uchar	padd_offset_bottom,	// ((layer_config[j][weight_m]-layer_config_original[j][weight_m])-padd_offset_top
	// Data Ports
	__global DPTYPE	*restrict	top,
	__global DPTYPE *restrict	top_location
	)
{
	ushort global_x = get_global_id(0); // max value 256
	ushort global_y = get_global_id(1); // max value 256
	ushort global_z = get_global_id(2); // max value 4096
	uchar  local_x 	= get_local_id(0); 	// max value 256
	uchar  local_y 	= get_local_id(1); 	// max value 256
	uchar  local_z 	= get_local_id(2); 	// max value 256
	uchar  index_z_item; 				// max value 256
	uchar  index_z_item_location; 		// max value 256
	ushort index_z_group;				// max value 4096
	ushort index_z_group_location;		// max value 4096
	channel_scal		output;
	channel_scal		output_location;
	__local DPTYPE		buffer[LANE_NUM];
	__local DPTYPE		buffer_location[LANE_NUM];

	if(local_z==0){
		if((control&0x02)==0x02){
			output = read_channel_intel(unpooling_ch);
		}
		else if((control&0x03)==0x01){
			output = read_channel_intel(pooling_ch);
			output_location = read_channel_intel(pooling_ch_location);
		}
		else
			output = read_channel_intel(bypass_ch);
		#pragma unroll
		for(uchar ll=0; ll<LANE_NUM; ll++){
			if((control&0x03)==0x01){
			 	#pragma unroll
			  	for(uchar ll=0; ll<LANE_NUM; ll++)
		    	buffer_location[ll]=output_location.lane[ll];
			}
		  	buffer[ll]=output.lane[ll];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// fetch data from local buffer and write back to DDR
	index_z_group = (global_z-padd_offset_top)/VEC_SIZE;
	index_z_item  = (global_z-padd_offset_top)%VEC_SIZE;
	index_z_group_location = (global_z-padd_offset_top)/LANE_NUM;
	index_z_item_location = (global_z-padd_offset_top)%LANE_NUM;
	
	if(global_z<(out_dim3-padd_offset_bottom) && (global_z>=padd_offset_top)) {
		top[index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];		
		if(((control&0x03)==0x01))
			top_location[index_z_group_location*out_dim1x2xbatch*LANE_NUM + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*LANE_NUM + (global_x+batch_indx_dim1*out_dim1)*LANE_NUM + index_z_item_location] = buffer_location[local_z];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}
