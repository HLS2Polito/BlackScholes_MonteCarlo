/* 
======================================================
 Copyright 2016 Liang Ma

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
======================================================
*
* Author:   Liang Ma (liang-ma@polito.it)
*
*  NUM_RNGS is the number of RNG objects implemented
*  NUM_SIMGROUPS 	is the number simulation groups (it must be an even number)
*  NUM_STEPS	is the number of time steps/partitions, which is 128 for the Asian option.
*
* init_array(), a static member function of RNG is defined here due to the dimension of the array.
*
* The algorithm in sampleSIM() is optimized from the mathematical and FPGA 
* implementation points of view for the Asian option.
*
*----------------------------------------------------------------------------
*/
#include "../common/blackScholes.h"

const int blackScholes::NUM_RNGS=2;
#ifdef LONG_SIMULATION
const int blackScholes::NUM_SIMS=512;
#else
const int blackScholes::NUM_SIMS=64;
#endif
const int blackScholes::NUM_SIMGROUPS=2;
const int blackScholes::NUM_STEPS=128;

blackScholes::blackScholes(stockData data):data(data)
{
}
void blackScholes::sampleSIM(RNG* mt_rng, data_t *pCall, data_t *pPut)
{
	const data_t Dt=data.timeT/data_t(NUM_STEPS);
	const data_t ratio1=expf(-data.freeRate*data.timeT);
	const data_t ratio2=(data.freeRate-data.volatility*data.volatility*0.5f)*Dt;
	const data_t ratio3=Dt*data.volatility*data.volatility;
	const data_t initValue=logf(data.initPrice/NUM_STEPS/1.0f);
	data_t call=0.0f,put=0.0f;

	data_t sum_call[NUM_RNGS], sum_put[NUM_RNGS];
#pragma HLS ARRAY_PARTITION variable=sum_call complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_put complete dim=1

	data_t sum_call1[NUM_RNGS], sum_put1[NUM_RNGS];
#pragma HLS ARRAY_PARTITION variable=sum_call1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_put1 complete dim=1

	data_t acc1[NUM_RNGS][NUM_SIMS],acc[NUM_RNGS][NUM_SIMS];
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

	data_t ave1[NUM_RNGS][NUM_SIMS],ave[NUM_RNGS][NUM_SIMS];
#pragma HLS ARRAY_PARTITION variable=ave1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ave complete dim=1
	data_t num1[NUM_RNGS][NUM_SIMS],num2[NUM_RNGS][NUM_SIMS];
#pragma HLS ARRAY_PARTITION variable=num2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=num1 complete dim=1

	loop_set:for(int k=0;k<NUM_SIMS;k++) {
#pragma HLS PIPELINE
		loop_set1:for(int i=0;i<NUM_RNGS;i++) {
	#pragma HLS UNROLL
			if(k==0) {
				sum_call[i]=0.0f;
				sum_put[i]=0.0f;
				sum_call1[i]=0.0f;
				sum_put1[i]=0.0f;
			}
			acc1[i][k]=initValue;
			ave1[i][k]=-data.strikePrice;
			acc[i][k]=initValue;
			ave[i][k]=-data.strikePrice;
		}
	}
	loop_main:for(int s=0;s<NUM_SIMGROUPS;s+=2) {
		loop_sim:for(int p=0;p<NUM_STEPS;p++) {
			loop_share:for(int k=0;k<NUM_SIMS;k++) {
	#pragma HLS PIPELINE
				loop_parallel:for(int i=0;i<NUM_RNGS;i++) {
					mt_rng[i].BOX_MULLER(&num1[i][k],&num2[i][k],ratio3);

					acc1[i][k]+=ratio2+num1[i][k];
					ave1[i][k]+=expf(acc1[i][k]);

					acc[i][k]+=ratio2+num2[i][k];
					ave[i][k]+=expf(acc[i][k]);
				}
			}
		}
		loop_sumShare:for(int k=0;k<NUM_SIMS;k++) {
			loop_sum:for(int i=0;i<NUM_RNGS;i++) {
#pragma HLS UNROLL
			//	std::cout<<"i="<<i<<" k="<<k<<" ave="<<ave[i][k]<<" ave1="<<ave1[i][k]<<std::endl;
				if(ave[i][k]>0.0f)
					sum_call[i]+=ave[i][k];
				else
					sum_put[i]+=ave[i][k];
				if(ave1[i][k]>0.0f)
					sum_call1[i]+=ave1[i][k];
				else
					sum_put1[i]+=ave1[i][k];
				acc1[i][k]=initValue;
				ave1[i][k]=-data.strikePrice;
				acc[i][k]=initValue;
				ave[i][k]=-data.strikePrice;
			}
		}
	}
	loop_sum_all:for(uint i=0;i<NUM_RNGS;i++) {
			call+=sum_call[i]+sum_call1[i];
			put+=sum_put[i]+sum_put1[i];
	}
	*pCall=ratio1*call/(NUM_SIMS*NUM_RNGS*NUM_SIMGROUPS*1.0f);
	*pPut =-ratio1*put/(NUM_SIMS*NUM_RNGS*NUM_SIMGROUPS*1.0f);
}

void blackScholes::simulation(data_t *call, data_t *put)
{

	RNG mt_rng[NUM_RNGS];
#pragma HLS ARRAY_PARTITION variable=mt_rng complete dim=1

	uint seeds[NUM_RNGS];
#pragma HLS ARRAY_PARTITION variable=seeds complete dim=1

	loop_seed:for(int i=0;i<NUM_RNGS;i++)
	{
#pragma HLS UNROLL
		seeds[i]=i;
	}
	RNG::init_array(mt_rng,seeds,NUM_RNGS);
	sampleSIM(mt_rng,call,put);

}
