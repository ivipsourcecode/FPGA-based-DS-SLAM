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

#ifndef _HW_PARAM_H
#define _HW_PARAM_H

// Macro architecture parameters

//// General
#define VEC_SIZE			            32		// larger than 4, i.e., 4, 8, 16 ...
#define LANE_NUM			            32		// larger than 1, i.e., 2, 4, 8, 16 ...
#define CHN_DEPTH			            0
#define LOC_MAX_BUF_SIZE	            512	// 2^9=512>480

//// Kernel memRd
#define WEIGHT_W                        3
#define WEIGHT_W_BANK_SIZE              4
#define WEIGHT_H                        3
#define CONV_GP_SIZE_X		            7
#define CONV_GP_SIZE_Y                  1		            // In this version, CONV_GP_SIZE_Y must be 1
#define WIN_BUF_SIZE                    16*4*512/VEC_SIZE   // >((CONV_GP_SIZE_X-1)*stride+7)*7*64/VEC_SIZE, for SegNet  batch=1
#define WEIGHT_BUF_SIZE                 4608/VEC_SIZE       // for SegNet  batch=1
#define MAX_WEIGHT_LAYER_BUF_SIZE       4*512/VEC_SIZE      // >7*64*64
#define WIN_BANK_BIT                    13                  // [log2(WIN_BUF_SIZE)]
#define WEIGHT_BANK_BIT                 11                  // [log2(MAX_WEIGHT_LAYER_BUF_SIZE)]
#define BIAS_BUF_SIZE                   512

//// Kernel conv
#define PIPE_DEPTH                      3
#define KERNEL_SIZE                     3
//// Kernel pool
#define POOL_MAX_SIZE                   2

//// Kernel unpooling
#define LINE_MAX_SIZE                   256

// Parameters for fixed-point design
#define CZERO       		            0x00     	// constant zero
#define CZERO_32			            0x00000000	// constant zero
#define MASK2B      		            0x03    	// used for final rounding
#define MASK8B      		            0xFF     	// used for final rounding
#define MASK9B      		            0x1FE    	// used for final rounding
#define MASK16B      		            0xFFFF    	// used for final rounding
#define MASKSIGN    		            0x80     	// used for final rounding
#define MASK_ACCUM  		            0xFFFFFFFF 	// use this value
#define MASK_MULT   		            0xFFFFFFFF 	// use this value

#endif

