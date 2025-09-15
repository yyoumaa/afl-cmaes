#include <stddef.h> /* NULL */
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include <math.h>
#include <string.h> /* strncmp */
#include "cmaes_interface.h"
#define operator_num 16
double stage_finds_score_all[operator_num] = {0.0};
double operator_prob[operator_num];
double sigma[operator_num];
double probability_now[operator_num] = {0.0};
FILE *fp;
cmaes_t evo; /* the optimizer */
double *arFunvals; /* objective function values of sampled population */



double fitfun(const double *y, int N) {
  double max_y = y[0], sum_exp = 0.0;
  for (int i = 1; i < N; ++i) if (y[i] > max_y) max_y = y[i];
  for (int i = 0; i < N; ++i)
    sum_exp += exp(y[i] - max_y);

  double weighted_score = 0.0;
  for (int i = 0; i < N; ++i) {
    double pi = exp(y[i] - max_y) / sum_exp;
    weighted_score += pi * stage_finds_score_all[i];//使用累计增益更新
  }

  return -weighted_score;
}

void cma_updating(void) {
    // 只更新一代，而不是运行到收敛
    double *const* pop = cmaes_SamplePopulation(&evo);
    for (int i = 0; i < cmaes_Get(&evo, "lambda"); ++i)
        arFunvals[i] = fitfun(pop[i], operator_num);
    cmaes_UpdateDistribution(&evo, arFunvals);

    // 获取当前均值（不要用GetNew，用GetPtr避免内存分配）
    const double *xopt = cmaes_GetPtr(&evo, "xmean");
    fprintf(fp, "\n------cma_updating------\n\n operator_prob: ");
    double sum_exp = 0.0;
    double max_x = xopt[0];
    for (int i = 1; i < operator_num; ++i)
        if (xopt[i] > max_x) max_x = xopt[i];
    
    for (int i = 0; i < operator_num; ++i)
        sum_exp += exp(xopt[i] - max_x);  // 数值稳定
    
    for (int i = 0; i < operator_num; ++i) {
        operator_prob[i] = exp(xopt[i] - max_x) / sum_exp;
        fprintf(fp, "%lf ", operator_prob[i]);
    }
    // while (!cmaes_TestForTermination(&evo)) {
    //   double *const* pop = cmaes_SamplePopulation(&evo);
    //   for (int i = 0; i < cmaes_Get(&evo, "lambda"); ++i)
    //     arFunvals[i] = fitfun(pop[i], operator_num);
    //   cmaes_UpdateDistribution(&evo, arFunvals);
    // }

    // fprintf(fp,"\n------cma_updating------\n\n operator_prob: ");
    // // const double *xopt = cmaes_GetNew(&evo, "xmean");
    // double sum_exp= 0.0;
    // for (int i = 0; i < operator_num; ++i)
    //   sum_exp += exp(xopt[i]);
    // for (int i = 0; i < operator_num; ++i){
    //   operator_prob[i] = exp(xopt[i]) / sum_exp;
    //   fprintf(fp,"%lf ",operator_prob[i]);
    // }
      
    // free((void*)xopt);
}

int main(){
    fp = fopen("/cma-log/cmaes.log-demo", "w");
    if (fp == NULL)
    {
      exit(1);
    }
   srand((unsigned)time(NULL)); 
    fprintf(fp,"init operator_prob: ");

    double total_operator_prob=0.0;
    for (int i = 0; i < operator_num; ++i) {
      double value=0.5;
      operator_prob[i] =value;    // 例如均匀或已有分布
      sigma[i] = 0.01;                 // 初始标准差
      total_operator_prob+=value;
      fprintf(fp,"%lf ",operator_prob[i]);
    }
    fprintf(fp,"\n");

    //归一化算子概率
    fprintf(fp,"init operator_prob after normlize: ");
    for (int i = 0; i < operator_num; i++) {  
      operator_prob[i] = operator_prob[i] / total_operator_prob;
      fprintf(fp,"%lf ",operator_prob[i]);
    }
    fprintf(fp,"\n");

    //计算累计概率用于抽样
    fprintf(fp,"probability_now: ");
    for (int i = 0; i < operator_num; i++)
		{
			if (i != 0)
				probability_now[i] = probability_now[i - 1] + operator_prob[i];
			else
				probability_now[i] = operator_prob[i];
      fprintf(fp,"%lf ",probability_now[i]);
		}
    fprintf(fp,"\n");

    arFunvals = (cmaes_init)(&evo, operator_num, operator_prob, sigma, 0, 0, NULL);

    fprintf(fp,"\n %s\n", cmaes_SayHello(&evo));
    
    for (int i = 0; i < operator_num; i++)
      {
          stage_finds_score_all[i]=0.0;
      }

      for(int i=0;i<100;i++){
        //随机选中n个算子增加n分  模拟实际fuzz情况
        
        int cishu = rand() % 100 + 1;
        for(int j=0;j<cishu;j++){
          int random_index = rand() % operator_num; //随机选中一个算子
          stage_finds_score_all[random_index] += 10.0; //增加分数
        }
        fprintf(fp,"\nstage_finds_score_all: ");
        for(int j=0;j<operator_num;j++){
          fprintf(fp,"%lf ",stage_finds_score_all[j]);
        }
        fprintf(fp,"\n");
        cma_updating(); //更新算子概率
        fprintf(fp,"\n after cma_updating probability_now: ");

      for (int i = 0; i < operator_num; i++)
      {
        if (i != 0)
          probability_now[i] = probability_now[i - 1] + operator_prob[i];
        else
          probability_now[i] = operator_prob[i];
        fprintf(fp,"%lf ",probability_now[i]);
      }
      fprintf(fp,"\n");
      }
    return 0;
}