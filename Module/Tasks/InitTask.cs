using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace EvolutionModule.Tasks
{
    /// <summary>
    /// Task for initializing evolutionary search.
    /// </summary>
    [Description("InitTask"), MyTaskInfo(OneShot = true)]
    public class InitTask : MyTask<EvolutionNode>
    {
        //private MyCudaKernel utilityKernel;

        public override void Init(int nGPU)
        {
            //utilityKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\UtilitiesKernel", "UtilitiesKernel");
            //utilityKernel.SetupExecution(Owner.InnerPopulationSize);
        }

        public override void Execute()
        {

            Owner.outerSteps = 0;
            Owner.innerSteps = 0;
            Owner.sampleSteps = 0;

            //utilityKernel.Run(Owner.Utility, (float)Owner.InnerPopulationSize);

            float total = 0;
            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                Owner.Utility.Host[i] = (float)Math.Max(0, Math.Log((float)Owner.InnerPopulationSize / 2 + 1) - Math.Log(i+1));
                total += Owner.Utility.Host[i];
                //Console.Write(Owner.Utility.Host[i] + " ");
            }

            //Console.WriteLine();
            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                Owner.Utility.Host[i] = Owner.Utility.Host[i] / total - 1f / Owner.InnerPopulationSize;
                //Console.Write(Owner.Utility.Host[i] + " ");
            }
            //Console.WriteLine();
            Owner.Utility.SafeCopyToDevice();


            float normalization = 0;
            for (int i = 0; i < Owner.AllCombinations; i++)
            {
                Owner.PopulationDistribution.Host[i] = (float)1 / (i + 1);
                normalization += Owner.PopulationDistribution.Host[i];
            }


            for (int i = 0; i < Owner.AllCombinations; i++)
            {
                Owner.PopulationDistribution.Host[i] = Owner.PopulationDistribution.Host[i] / normalization;
                Owner.PopulationFitnesses.Host[i] = 0.5f;
            }

            Owner.PopulationFitnesses.SafeCopyToDevice();
            Owner.PopulationDistribution.SafeCopyToDevice();

            Owner.Means.Fill(0);

            for (int i = 0; i < Owner.AllCombinations * Owner.MaxNumberOfCoefficients; i++)
            {
                Owner.Sigmas.Host[i] = Owner.StartSigma;
            }
            Owner.Sigmas.SafeCopyToDevice();

            int steps = Math.Max((Owner.MaxNumberOfCoefficients - Owner.MinNumberOfCoefficients)
                / Owner.CoefficientSteps, 1);
            Debug.WriteLine("steps " + steps);
            int oneRun = Math.Max((int)(Owner.MaxNumberOfWeights - Owner.MinNumberOfWeights)
                    / Owner.WeightSteps, 1);
            Debug.WriteLine("oneRun " + oneRun);
            int nrOfCoefficients = Owner.MinNumberOfCoefficients;
            int nrOfWeights = Owner.MinNumberOfWeights;


            for (int i = 0; i < steps; i++)
            {
                for (int j = 0; j < oneRun; j++)
                {
                    Owner.NumberOfWeights.Host[i * oneRun + j] = nrOfWeights;
                    Owner.NumberOfCoefficients.Host[i * oneRun + j] = nrOfCoefficients;
                    nrOfWeights += Owner.WeightSteps;
                    //Console.WriteLine(Owner.Means.Host[(i * oneRun + j) * Owner.MaxNumberOfCoefficients] + " "
                    //    + Owner.Sigmas.Host[(i * oneRun + j) * Owner.MaxNumberOfCoefficients]);
                }
                nrOfWeights = Owner.MinNumberOfWeights;
                nrOfCoefficients += Owner.CoefficientSteps;
            }

            Owner.NumberOfCoefficients.SafeCopyToDevice();
            Owner.NumberOfWeights.SafeCopyToDevice();


            for (int i = 0; i < Owner.AllCombinations; i++)
            {
                Owner.NotConverged.Host[i] = 1;
            }
            Owner.NotConverged.SafeCopyToDevice();
        }
    }
}
