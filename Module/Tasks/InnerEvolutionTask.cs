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
    /// Inner evolution
    /// In one in every InnerPopulationSize SimulationSteps 
    /// - new network is created from coefficients
    /// In one in every (InnerPopulationSize x NumberOfIterations) SimulationSteps 
    /// - coefficient distributions are updated.
    /// In one in every (InnerEvolutionSteps x InnerPopulationSize x NumberOfIterations) SimulationSteps 
    /// - new complexity class = number of coefficients + number of weights is drawn.
    /// </summary>
    [Description("Inner Evolution Task")]
    public class InnerEvolutionTask : MyTask<EvolutionNode>
    {
        [MyBrowsable, Category("Learning rate"), YAXSerializableField(DefaultValue = 1f)]
        public float LearningRateMean { get; set; }
        [MyBrowsable, Category("Learning rate"), YAXSerializableField(DefaultValue = 0.05f)]
        public float LearningRateSigma { get; set; }

        //Switch between different reinforcement learning tasks
        [MyBrowsable, Category("Task"), YAXSerializableField(DefaultValue = true)]
        public bool agent { get; set; }
        [MyBrowsable, Category("Task"), YAXSerializableField(DefaultValue = false)]
        public bool grid { get; set; }
        [MyBrowsable, Category("Evolution"), YAXSerializableField(DefaultValue = false)]
        public bool direct { get; set; }

        private MyCudaKernel discriteCosineTransform;

        string myreward = "";

        public float[,] Sk;
        public float[,] Zk;

        int coefficientLength;
        int numberOfWeights;

        My2DAgentWorld agentWorld;
        MyGridWorld gridworld;

        String file;

        public override void Init(int nGPU)
        {
            int ending = 1;
            file = "C://Users//Alena Moravova//Desktop//inner_fitness_" + ending + ".csv";
            while (File.Exists(file))
            {
                ending++;
                file = "C://Users//Alena Moravova//Desktop//inner_fitness" + ending + ".csv";
            }

            discriteCosineTransform = MyKernelFactory.Instance.Kernel(nGPU, @"\DCTKernel", "DiscriteCosineTransform");
            discriteCosineTransform.SetupExecution(Owner.MaxNumberOfCoefficients * Owner.MaxNumberOfWeights);
            discriteCosineTransform.DynamicSharedMemory = sizeof(float) * (uint)Owner.MaxNumberOfCoefficients;


            if (agent)
                agentWorld = (My2DAgentWorld)Owner.Owner.World;
            else if (grid)
                gridworld = (MyGridWorld)Owner.Owner.World;

        }


        /// <summary>
        /// Executed at each SimulationStep. 
        /// Method's execution parts are divided according to
        /// the SimulationSteps and number of samples drawn for
        /// outer evolution and number of networks drawn for one sample.
        /// </summary>
        public override void Execute()
        {
            if (SimulationStep == 0)
            {
                DrawNewSample();

                ////adds sample number to the output
                myreward = Owner.sampleIndex + ",";

                PrepareCoefficients();
                PrepareNetwork();
            }

            // Calls method for generating new distribution for inner population.
            // Draws new complexity class for evolution.
            if (((SimulationStep - 1) % Owner.NumberOfIterations) == 0 && (SimulationStep - 1) != 0)
            {
                //end of one run of inner evolution for one sample
                if (Owner.innerSteps == (Owner.InnerPopulationSize - 1))
                {
                    UpdateDistributionsOfSample();

                    //end of inner evolution of one sample
                    if (Owner.sampleSteps == (Owner.InnerEvolutionSteps - 1))
                    {
                        myreward = myreward + Owner.Fitness.Host[Owner.InnerPopulationSize - 1] + ",";

                        DrawNewSample();

                        //add sample number to the output
                        myreward = myreward + "\n";
                        File.AppendAllText(file, myreward);
                        myreward = Owner.sampleIndex + ",";

                        Owner.sampleSteps = 0;

                        //not the end of outer evolution of all samples
                        if (Owner.outerSteps == (Owner.OuterPopulationSize - 1))
                        {
                            Owner.outerSteps = 0;
                        }
                        else
                        {
                            Owner.outerSteps++;
                        }
                    }
                    else
                    {
                        myreward = myreward + Owner.Fitness.Host[Owner.InnerPopulationSize - 1] + ",";
                        Owner.sampleSteps++;
                    }

                    PrepareCoefficients();
                    Owner.innerSteps = 0;
                }
                else
                {
                    Owner.innerSteps++;
                }

                PrepareNetwork();

                //RESET WORLDS
                ResetWorlds();
            }
        }


        /// <summary>
        /// Reseting worlds for agent into the initial state.
        /// </summary>
        void ResetWorlds()
        {
            if (agent && (SimulationStep - 1) % (Owner.NumberOfIterations) == 0 && (SimulationStep - 1) != 0)
            {
                agentWorld.InitGameTask.Execute();
            }
            else if (grid && (SimulationStep - 1) % (Owner.NumberOfIterations) == 0 && (SimulationStep - 1) != 0)
            {
                gridworld.InitGameTask.Execute();
                gridworld.GlobalOutput.Host[2] = 1;
                gridworld.GlobalOutput.SafeCopyToDevice();
                gridworld.ResetAgentTask.Execute();
            }
        }

        /// <summary>
        /// Method draws new sample from the distribution. Every sample has
        /// assigned # of coefficients and # of weights, and for each coefficient
        /// its mean and sigma for creating it's values.
        /// </summary>
        void DrawNewSample()
        {
            Random rand = new Random();
            double randnumber = rand.NextDouble();
            double sum = 0;

            for (int i = 0; i < Owner.AllCombinations; i++)
            {
                sum += Owner.PopulationDistribution.Host[i];
                if (randnumber < sum)
                {
                    Owner.sampleIndex = i;
                    break;
                }
            }
            //Console.WriteLine();
            //Console.WriteLine("coefficients " + Owner.NumberOfCoefficients.Host[sampleIndex]);
            //Console.WriteLine("weights " + Owner.NumberOfWeights.Host[sampleIndex]);
        }


        /// <summary>
        /// Method updates the means and sigmas for every coefficient for 
        /// evaluated sample. The natural gradient update is used.
        /// </summary>
        void UpdateDistributionsOfSample()
        {
            //find best settings for this complexity
            int[] indexes = new int[Owner.InnerPopulationSize];
            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                indexes[i] = i;
            }

            Array.Sort(Owner.Fitness.Host, indexes);
            Array.Reverse(indexes);

            float MeanDelta = 0;
            float SigmaDelta = 0;

            Owner.NotConverged.Host[Owner.sampleIndex] = 0;

            int offset = Owner.sampleIndex * Owner.MaxNumberOfCoefficients;

            //for (int i = 0; i < Owner.InnerPopulationSize; i++)
            //{
            //    Console.WriteLine(Owner.Fitness.Host[Owner.InnerPopulationSize - 1 - i] + " utility: " + Owner.Utility.Host[i]);
            //}

            for (int j = 0; j < coefficientLength; j++)
            {
                MeanDelta = 0;
                SigmaDelta = 0;
                for (int i = 0; i < Owner.InnerPopulationSize; i++)
                {
                    int indexI = indexes[i];
                    MeanDelta += Owner.Utility.Host[i] * Sk[indexI, j];
                    SigmaDelta += Owner.Utility.Host[i] * (Sk[indexI, j] * Sk[indexI, j] - 1);
                }

                Owner.Means.Host[offset + j] = Owner.Means.Host[offset + j] + LearningRateMean * MeanDelta * Owner.Sigmas.Host[offset + j];
                Owner.Sigmas.Host[offset + j] = Owner.Sigmas.Host[offset + j] * (float)Math.Exp(LearningRateSigma * SigmaDelta);

                // if any variance > threshold
                if (Owner.Sigmas.Host[offset + j] > Owner.threshold)
                {
                    Owner.NotConverged.Host[Owner.sampleIndex] = 1;
                }
            }

            Owner.NotConverged.SafeCopyToDevice();
            if (Owner.PopulationFitnesses.Host[Owner.sampleIndex] < Owner.Fitness.Host[Owner.InnerPopulationSize - 1])
                Owner.PopulationFitnesses.Host[Owner.sampleIndex] = Owner.Fitness.Host[Owner.InnerPopulationSize - 1];

            //Console.WriteLine("reward " + Owner.PopulationFitnesses.Host[sampleIndex]);
            Owner.PopulationFitnesses.SafeCopyToDevice();

            Owner.Sigmas.SafeCopyToDevice();
            Owner.Means.SafeCopyToDevice();
        }


        /// <summary>
        /// Coefficients for a sample are generated from the gaussian 
        /// distribution associated with each of them.
        /// </summary>
        public void PrepareCoefficients()
        {
            // prepare for next network
            coefficientLength = (int)Owner.NumberOfCoefficients.Host[Owner.sampleIndex];
            numberOfWeights = (int)Owner.NumberOfWeights.Host[Owner.sampleIndex];

            Sk = new float[Owner.InnerPopulationSize, coefficientLength];
            Zk = new float[Owner.InnerPopulationSize, coefficientLength];

            float D = (float)Math.Pow(Owner.INPUT_UNITS + Owner.OUTPUT_UNITS, 2) + 4 * numberOfWeights;
            Owner.hiddenLayerSize = Math.Max(1, (-(Owner.INPUT_UNITS + Owner.OUTPUT_UNITS) + (int)Math.Sqrt(D)) / 2);

            Random rand = new Random();
            for (int i = 0; i < Owner.InnerPopulationSize; i++)
            {
                for (int j = 0; j < coefficientLength; j++)
                {
                    double u1 = rand.NextDouble(); //uniform(0,1) random doubles
                    double u2 = rand.NextDouble();

                    double randNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                    double randNormalZk =
                                 Owner.Means.Host[Owner.sampleIndex + j] + Owner.Sigmas.Host[Owner.sampleIndex + j] * randNormal; //random normal(mean,stdDev^2)
                    Sk[i, j] = (float)randNormal;
                    Zk[i, j] = (float)randNormalZk;
                }
            }
        }


        /// <summary>
        /// Method runs the DCTkernel which transforms the 
        /// coefficients into the weights of a network.
        /// </summary>
        public void PrepareNetwork()
        {
            //reseting networks' internal states
            Owner.HiddenActivations.Fill(0);
            Owner.PreviousHiddenActivations.Fill(0);
            Owner.InputWeights.Fill(0);
            Owner.RecurrentWeights.Fill(0);
            Owner.OutputWeights.Fill(0);
            Owner.InnerEvolWeights.Fill(0);
            Owner.InnerEvolCoefficients.Fill(0);
            Owner.DCTMatrix.Fill(0);

            for (int i = 0; i < coefficientLength; i++)
            {
                Owner.InnerEvolCoefficients.Host[i] = Zk[Owner.innerSteps, i];
            }
            Owner.InnerEvolCoefficients.SafeCopyToDevice();

            if (!direct)
            {
                discriteCosineTransform.Run(
                    Owner.InnerEvolCoefficients,
                    Owner.DCTMatrix,
                    Owner.InnerEvolWeights,
                    this.coefficientLength,
                    this.numberOfWeights);
            }
            else
            {
                for (int i = 0; i < coefficientLength; i++)
                {
                    Owner.InnerEvolWeights.Host[i] = Owner.InnerEvolCoefficients.Host[i];
                }
                Owner.InnerEvolWeights.SafeCopyToDevice();
            }

            Owner.InnerEvolWeights.CopyToMemoryBlock(
                Owner.InputWeights,
                0,
                0,
                Owner.INPUT_UNITS * Owner.hiddenLayerSize);

            Owner.InnerEvolWeights.CopyToMemoryBlock(
                Owner.RecurrentWeights,
                Owner.INPUT_UNITS * Owner.hiddenLayerSize,
                0,
                Owner.hiddenLayerSize * Owner.hiddenLayerSize);

            Owner.InnerEvolWeights.CopyToMemoryBlock(
                Owner.OutputWeights,
                Owner.INPUT_UNITS * Owner.hiddenLayerSize + Owner.hiddenLayerSize * Owner.hiddenLayerSize,
                0,
                Owner.OUTPUT_UNITS * Owner.hiddenLayerSize);
        }

    }
}



