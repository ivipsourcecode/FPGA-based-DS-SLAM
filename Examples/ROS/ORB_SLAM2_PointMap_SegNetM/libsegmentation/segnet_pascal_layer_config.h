#ifndef SEGNET_PASCAL_LAYER_CONFIG.H  
#define SEGNET_PASCAL_LAYER_CONFIG.H 

#include "./device/segnet_hw_param666_W.cl"

#define NUM_CONFIG_ITEM  33

// SegNet11 Configuration
extern unsigned layer_config[][NUM_CONFIG_ITEM] = {
	{224, 224, 3, 3, 3, 3, 64, 64, 64, 64,
	0,
	0,
	224, 224, 64, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			// "0" -> Don't pool
	0, 0, 0, 0, 0, 0,				// "0" -> Don't uppooling
	1,
	0},//Layer-con1_1
	{224, 224, 64, 3, 3, 64, 64, 64, 64, 64,
	1,
	0,
	224, 224, 64, 1, 1, 0, 1,
	1, 112, 112, 64, 2, 2,			
	0, 0, 0, 0, 0, 0,				
	0,
	1},//Layer-1_2
	{112, 112, 64, 3, 3, 64, 128, 128, 128, 128,
	0,
	0,
	112, 112, 128, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-2_1
	{112, 112, 128, 3, 3, 128, 128, 128, 128, 128,
	1,
	0,
	112, 112, 128, 1, 1, 0, 1,
	1, 56, 56, 128, 2, 2,
	0, 0, 0, 0, 0, 0,				
	0,
	2},//Layer-2_2
	{56, 56, 128, 3, 3, 128, 256, 256, 256, 256,
	0,
	0,
	56, 56, 256, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-3_1
	{56, 56, 256, 3, 3, 256, 256, 256, 256, 256,
	1,
	0,
	56, 56, 256, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-3_2
	{56, 56, 256, 3, 3, 256, 256, 256, 256, 256,
	0,
	0,
	56, 56, 256, 1, 1, 0, 1,
	1, 28, 28, 256, 2, 2,
	0, 0, 0, 0, 0, 0,				
	1,
	3},//Layer-3_3
	{28, 28, 256, 3, 3, 256, 512, 512, 512, 512,
	1,
	0,
	28, 28, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-4_1
	{28, 28, 512, 3, 3, 512, 512, 512, 512, 512,
	0,
	0,
	28, 28, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-4_2
	{28, 28, 512, 3, 3, 512, 512, 512, 512, 512,
	1,
	0,
	28, 28, 512, 1, 1, 0, 1,
	1, 14, 14, 512, 2, 2,
	0, 0, 0, 0, 0, 0,				
	0,
	4},//Layer-4_3
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	0,
	0,
	14, 14, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-5_1
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	1,
	0,
	14, 14, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-5_2
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	0,
	0,
	14, 14, 512, 1, 1, 0, 1,
	1, 7, 7, 512, 2, 2,
	1, 14, 14, 512, 0, 0,
	1,
	0},//Layer-5_3
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	1,
	0,
	14, 14, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-5_3D
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	0,
	0,
	14, 14, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-5_2D
	{14, 14, 512, 3, 3, 512, 512, 512, 512, 512,
	1,
	4,
	14, 14, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	1, 28, 28, 512, 0, 0,
	0,
	0},//Layer-5_1D
	{28, 28, 512, 3, 3, 512, 512, 512, 512, 512,
	0,
	0,
	28, 28, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-4_3D
	{28, 28, 512, 3, 3, 512, 512, 512, 512, 512,
	1,
	0,
	28, 28, 512, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-4_2D
	{28, 28, 512, 3, 3, 512, 256, 256, 256, 256,
	0,
	3,
	28, 28, 256, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	1, 56, 56, 256, 0, 0,
	1,
	0},//Layer-4_1D
	{56, 56, 256, 3, 3, 256, 256, 256, 256, 256,
	1,
	0,
	56, 56, 256, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-3_3D
	{56, 56, 256, 3, 3, 256, 256, 256, 256, 256,
	0,
	0,
	56, 56, 256, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-3_2D
	{56, 56, 256, 3, 3, 256, 128, 128, 128, 128,
	1,
	2,
	56, 56, 128, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	1, 112, 112, 128, 0, 0,
	0,
	0},//Layer-3_1D
	{112, 112, 128, 3, 3, 128, 128, 128, 128, 128,
	0,
	0,
	112, 112, 128, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-2_2D
	{112, 112, 128, 3, 3, 128, 64, 64, 64, 64,
	1,
	1,
	112, 112, 64, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	1, 224, 224, 64, 0, 0,
	0,
	0},//Layer-2_1D
	{224, 224, 64, 3, 3, 64, 64, 64, 64, 64,
	0,
	0,
	224, 224, 64, 1, 1, 0, 1,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	1,
	0},//Layer-1_2D
	{224, 224, 64, 3, 3, 64, 21, 21, 21, 21,
	1,
	0,
	224, 224, 21, 1, 1, 0, 0,
	0, 0, 0, 0, 0, 0,			
	0, 0, 0, 0, 0, 0,				
	0,
	0},//Layer-1_1D
};

			
extern char precision_config[][2] ={	
	{0,4},//Layer-1_1
	{4,4},//Layer-1_2
	{4,4},//Layer-2_1
	{4,3},//Layer-2_2
	{3,3},//Layer-3_1
	{3,3},//Layer-3_2
	{3,3},//Layer-3_3
	{3,3},//Layer-4_1
	{3,3},//Layer-4_2
	{3,3},//Layer-4_3
	{3,3},//Layer-5_1
	{3,3},//Layer-5_2
	{3,3},//Layer-5_3
	{3,3},//Layer-5_3D
	{3,3},//Layer-5_2D
	{3,4},//Layer-5_1D
	{4,4},//Layer-4_3D
	{4,3},//Layer-4_2D
	{3,3},//Layer-4_1D
	{3,3},//Layer-3_3D
	{3,4},//Layer-3_2D
	{4,4},//Layer-3_1D
	{4,4},//Layer-2_2D
	{4,4},//Layer-2_1D
	{4,3},//Layer-1_2D
	{3,1},//Layer-1_1D
};

extern unsigned input_config[4] 	= {224, 224, 3, 1}; //original image size(dim1, dim2, dim3), batch size
extern unsigned output_config[3] 	= {224, 224, 21};	//Layer-8  Note: only one result is extracted and verified

#endif
