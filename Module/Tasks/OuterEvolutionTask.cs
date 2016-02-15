using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.GameBoy;
using GoodAI.Modules.GridWorld;
using GoodAI.Modules.Matrix;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace EvolutionModule.Tasks
{
    /// <summary>
    /// Calculating new distribution for complexity classes after
    /// all drawn samples have been evaluated in inner evolution.
    /// </summary>
    [Description("Outer Evolution Task")]
    public class OuterEvolutionTask : MyTask<EvolutionNode>
    {
        private MyCudaKernel generateNewDistribution;

        String filePathFitness;
        String filePathDistribution;

        public override void Init(int nGPU)
        {
            int ending = 1;
            filePathFitness = "C://Users//Alena Moravova//Desktop//fitness_pop_" + ending + ".csv";
            while (File.Exists(filePathFitness))
            {
                ending++;
                filePathFitness = "C://Users//Alena Moravova//Desktop//fitness_pop_" + ending + ".csv";
            } 
            ending = 1;
            filePathDistribution = "C://Users//Alena Moravova//Desktop//distrib_pop_" + ending + ".csv";
            while (File.Exists(filePathDistribution))
            {
                ending++;
                filePathDistribution = "C://Users//Alena Moravova//Desktop//distrib_pop_" + ending + ".csv";
            }

            generateNewDistribution = MyKernelFactory.Instance.Kernel(nGPU, @"\PopulationDistributionKernel", "UpdateProbabilityDistribution");
            generateNewDistribution.SetupExecution(Owner.AllCombinations);
            generateNewDistribution.DynamicSharedMemory = sizeof(float) * (uint)Owner.AllCombinations;
        }

        public override void Execute()
        {
            if (SimulationStep == 0)
            {
                writeToFile();
            }
            // NumberOfSteps = number of steps before reading reward
            if (((SimulationStep - 1) % Owner.NumberOfIterations) == 0 && (SimulationStep - 1) != 0)
            {
                // allways checks reward
                Owner.Fitness.Host[Owner.innerSteps] = Owner.Reward.Host[0];
                Owner.Fitness.SafeCopyToDevice();
                //Console.WriteLine("fitness " + Owner.Fitness.Host[Owner.innerSteps]);

                // end of one step of outer evolution - all drawn samples evolved
                if (Owner.innerSteps == (Owner.InnerPopulationSize - 1) // end of evaluation of all drawn samples for inner evol.
                    && Owner.sampleSteps == (Owner.InnerEvolutionSteps - 1) // end of inner evolution
                    && Owner.outerSteps == (Owner.OuterPopulationSize - 1)) // end of outer evolution 
                {
                    // get best reward for sample
                    float maxReward= Owner.Fitness.Host.Max();
                    Owner.PopulationFitnesses.Host[Owner.sampleIndex] = maxReward;
                    //Console.WriteLine("fitness " + Owner.PopulationFitnesses.Host[Owner.sampleIndex]);
                    Owner.PopulationFitnesses.SafeCopyToDevice();

                    // get new evolution
                    OuterPopulationFinished();
                }
            }

        } 

        /// <summary>
        /// Generates new distribution for outer population.
        /// </summary>
        void OuterPopulationFinished()
        {

            if (Owner.AllCombinations > 1)
            {
                float total = 0f;

                //normalize fitnesses
                for (int i = 0; i < Owner.AllCombinations; i++)
                {
                    total += Owner.PopulationFitnesses.Host[i];
                }

                for (int i = 0; i < Owner.AllCombinations; i++)
                {
                    Owner.PopulationFitnesses.Host[i] /= total;
                }

                Owner.PopulationFitnesses.SafeCopyToDevice();
                Owner.PopulationDistribution.SafeCopyToDevice();


                generateNewDistribution.Run(Owner.NumberOfWeights, Owner.NumberOfCoefficients, Owner.PopulationFitnesses,
                    Owner.PopulationDistribution, Owner.NotConverged, Owner.AllCombinations, Owner.DistributionSmoothness);

                Owner.PopulationDistribution.SafeCopyToHost();
                
                
                //for (int i = 0; i < Owner.AllCombinations; i++)
                //{
                //    Console.WriteLine(Owner.PopulationDistribution.Host[i] + " " + Owner.PopulationFitnesses.Host[i]);
                //}
                //Console.WriteLine();

                writeToFile();
            }
        }

        void writeToFile()
        {
            var csv1 = new StringBuilder();
            var csv2 = new StringBuilder();

            var input1 = "";
            var input2 = "";

            for (int i = 0; i < Owner.AllCombinations; i++)
            {
                input1 = input1 + Owner.PopulationDistribution.Host[i] + ",";
                input2 = input2 + Owner.PopulationFitnesses.Host[i] + ",";
            }

            csv1.AppendLine(input1);
            csv2.AppendLine(input2);

            File.AppendAllText(filePathDistribution, csv1.ToString());
            File.AppendAllText(filePathFitness, csv2.ToString());
        }
    }
}
