/*
 * svmapp.c
 *
 *  Created on: 2019年3月27日
 *      Author: SpringZhang
 */

#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <float.h>

#define NSEC_PER_SEC		1000000000

struct period_info
{
	struct timespec next_period;
	long period_ns;
};
static inline int64_t calcdiff_ns(struct timespec t1, struct timespec t2)
{
	int64_t diff;
	//printf("start ns[%lu], ends ns [%lu]\n", (unsigned long) t2.tv_nsec,
			//(unsigned long) t1.tv_nsec);
	diff = NSEC_PER_SEC * (int64_t)((int) t1.tv_sec - (int) t2.tv_sec);
	diff += ((int) t1.tv_nsec - (int) t2.tv_nsec);
	return diff;
}

extern int scale_main(int argc, char **argv);
extern int train_main(int argc, char **argv);
double single_scaled(double value, const double y_min, const double y_max,const double y_lower, const double y_upper);

svm_node * get_svm_node(const double *rawdata, int rawdatalen);
const static int rawdataLen = 100;
//100组rawsensordata测试数据
double rawsensordata[rawdataLen] =
{ 25.961079, 27.003452, 28.043232, 29.07543, 30.097397, 31.109077, 32.11132,
		33.103767, 34.084038, 35.04871, 35.994911, 36.92144, 37.829311,
		38.721443, 39.601395, 40.472401, 41.336109, 42.191727, 43.03627,
		43.866291, 44.67944, 45.47509, 46.253975, 47.018036, 47.769699,
		48.511242, 49.244621, 49.971485, 50.693707, 51.412971, 52.130222,
		52.845634, 53.55859, 54.267822, 54.971165, 55.665741, 56.348293,
		57.01606, 57.666798, 58.299019, 58.911892, 59.504269, 60.074306,
		60.619377, 61.13628, 61.621563, 62.071907, 62.484497, 62.857929,
		63.192425, 63.489483, 63.751305, 63.980274, 64.178474, 64.347313,
		64.487793, 64.600716, 64.686615, 64.745743, 64.77803, 64.783569,
		64.762512, 64.715347, 64.643074, 64.547493, 64.430382, 64.293419,
		64.137566, 63.964066, 63.774906, 63.572453, 63.358898, 63.136589,
		62.908066, 62.675396, 62.440231, 62.203365, 61.965206, 61.726402,
		61.48772, 61.249729, 61.012257, 60.774429, 60.535324, 60.294315,
		60.05106, 59.805744, 59.558998, 59.312428, 59.067867, 58.826836,
		58.590565, 58.360184, 58.136787, 57.920792, 57.712032, 57.510208,
		57.315403, 57.128292, 56.949543 };
struct svm_node dynamic_rawdata[rawdataLen + 1] =
{ 0 };			//from sensor
/*svm_problem * get_svm_problem()
{

	svm_problem *prob = (struct svm_problem*) malloc(
			sizeof(struct svm_problem));
	prob->y = (double*) malloc(sizeof(double));
	prob->l = 1; //每次处理一行数据
	*(prob->y) = 0; //预设label 0

	struct svm_node* x_data = get_svm_node(rawsensordata, rawdataLen);
	prob->x = &x_data;

	//注意需要外部释放prob和x_data内存
	return prob;

 }*/

//根据原始测试数据建立svm_node数组
svm_node * get_svm_node(const double *rawdata, int rawdatalen)
{
	//struct svm_node* x_data = (struct svm_node*) malloc(
	//	(rawdatalen + 1) * sizeof(svm_node)); //样本数据元素空间比原始数据长度多一个结束符
	struct svm_node* x_data = dynamic_rawdata;
	for (int i = 0; i < rawdatalen; i++)
	{
		if (rawdata[i] != 0)
		{
			x_data[i].index = (i + 1);
			x_data[i].value = rawdata[i];
		}
	}
	x_data[rawdatalen].index = -1; //ending
	x_data[rawdatalen].value = 0;
	return x_data;
}
//generate predict model run once during app start up based on buffer data
/*svm_model* exo_svm_train()
{
	//scale the raw input data
	//define svm_parameter
	//prepare svm_problem
	//generate predict model
	svm_problem *svm_prob = get_svm_problem();
	svm_parameter *param = (struct svm_parameter*) malloc(
			sizeof(struct svm_parameter));
	param->svm_type = C_SVC;
	param->kernel_type = RBF;
	param->degree = 3;
	param->gamma = 0;	// 1/num_features
	param->coef0 = 0;
	param->nu = 0.5;
	param->cache_size = 100;
	param->C = 1;
	param->eps = 1e-3;
	param->p = 0.1;
	param->shrinking = 1;
	param->probability = 0;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;

	const char *error_msg = svm_check_parameter(svm_prob, param);
	if (error_msg)
	{
		printf("exo_svm_train svm_check_parameter ERROR: %s\n", error_msg);
		exit(1);
	}
	struct svm_model * svmmodel = svm_train(svm_prob, param);
	return svmmodel;
 }*/

struct svm_model *globalSVMModel;
const char * scale_output_file = "/tmp/scale.result";
const char * output_model_file = "scale.result.model";

//获取原始测试数据中的最大最小值
void get_minmax_value(double * rawdatainput, int rawdatalen, double *out_y_min,
		double *out_y_max)
{
	double y_min = DBL_MAX;
	double y_max = -DBL_MAX;
	for (int i = 0; i < rawdatalen; i++)
	{
		if (rawdatainput[i] < y_min)
			y_min = rawdatainput[i];
		if (rawdatainput[i] > y_max)
			y_max = rawdatainput[i];
	}
	*out_y_min = y_min;
	*out_y_max = y_max;
}
//对原始数据组进行缩放处理
int exo_svm_scale(double * rawdatainput, int rawdatalen, const double y_lower,
		const double y_upper)
{
	double y_min = 0;
	double y_max = 0;
	get_minmax_value(rawdatainput, rawdatalen, &y_min, &y_max);
	double scaledvalue = 0;
	for (int i = 0; i < rawdatalen; i++)
	{
		scaledvalue = single_scaled(rawdatainput[i], y_min, y_max, y_lower,
				y_upper);
		rawdatainput[i] = scaledvalue;
	}
	return 0;	//succeed
}

//对每单个数据进行缩放处理
double single_scaled(double value, const double y_min, const double y_max,
		const double y_lower, const double y_upper)
{
	if (y_min == y_max)
		return value;

	if (value == y_min)
		value = y_lower;
	else if (value == y_max)
		value = y_upper;
	else
		value = y_lower
				+ (y_upper - y_lower) * (value - y_min) / (y_max - y_min);

	return value;
}
//exo_svm_init run once during app start to generate the svm_model
//svm初始化：1，从文件读取训练数据；2，模型训练并返回模型参数
svm_model * exo_svm_init(const char * modelFile)
{
	const char* scaleargv[] =
	{ "svmmain", "train.data" };
	scale_main(2, (char**) scaleargv);

	char* trainargv[] =
	{ (char*) "svmmain", (char *) scale_output_file };
	train_main(2, trainargv);
	return svm_load_model(modelFile);
}
//对测试数据进行预测
double exo_gait_predict(const double *rawdata, int rawdatalen,
		const svm_model *model)
{
//1,load svm_model from file?
//2,prepare svm_node from sensor data
//3,double svm_predict(const svm_model *model, const svm_node *x);
	const svm_node* x_data = get_svm_node(rawdata, rawdatalen);
	double predictresult = svm_predict(model, x_data);
	return predictresult;
}
//数据打印输出
void dump_data(const double *rawdata, int rawdatalen)
{
	for (int i = 0; i < rawdatalen; i++)
	{
		printf(" %g ", rawdata[i]);
	}
	printf("\n");

}
int main()
{
	globalSVMModel = exo_svm_init(output_model_file);
	if (NULL == globalSVMModel)
	{
		printf(" failed to load svm model file\n");
		return -1;
	}

	while (1)
	{
		struct timespec startns;
		clock_gettime(CLOCK_MONOTONIC, &startns);
		printf("before scaled");
		dump_data(rawsensordata, rawdataLen);
		exo_svm_scale(rawsensordata, rawdataLen,-1,1);
		printf("after scaled");
		dump_data(rawsensordata, rawdataLen);

		double predictresult = exo_gait_predict(rawsensordata, rawdataLen,
				globalSVMModel);
		struct timespec endns;
		clock_gettime(CLOCK_MONOTONIC, &endns);
		printf("predict result is %.17g and real delay us [%lu]\n",
				predictresult,
				(unsigned long) (calcdiff_ns(endns, startns) / (1000)));
		usleep(100000);
	}
}


