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

        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.outerSteps = 0;
            Owner.innerSteps = 0;
            Owner.sampleSteps = 0;

            float total = 0;
            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                Owner.Utility.Host[i] = (float)Math.Max(0, Math.Log((float)Owner.InnerPopulationSize / 2 + 1) - Math.Log(i+1));
                total += Owner.Utility.Host[i];
            }

            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                Owner.Utility.Host[i] = Owner.Utility.Host[i] / total - 1f / Owner.InnerPopulationSize;
            }
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
            int oneRun = Math.Max((int)(Owner.MaxNumberOfWeights - Owner.MinNumberOfWeights)
                    / Owner.WeightSteps, 1);
            int nrOfCoefficients = Owner.MinNumberOfCoefficients;
            int nrOfWeights = Owner.MinNumberOfWeights;


            for (int i = 0; i < steps; i++)
            {
                for (int j = 0; j < oneRun; j++)
                {
                    Owner.NumberOfWeights.Host[i * oneRun + j] = nrOfWeights;
                    Owner.NumberOfCoefficients.Host[i * oneRun + j] = nrOfCoefficients;
                    nrOfWeights += Owner.WeightSteps;
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
