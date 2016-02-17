using System;
using System.ComponentModel;
using System.Linq;

using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using YAXLib;
using EvolutionModule.Tasks;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.Collections.Generic;

namespace EvolutionModule
{
    /// <author>Alena Moravova</author>
    /// <status>Working</status>
    /// <summary>Node executing evolution of neural networks.</summary>
    /// <description>
    /// Node that runs evolution of Neural networks in compressed weight space.
    /// </description>

    public class EvolutionNode : MyWorkingNode
    {
        /// <summary>
        /// Input from the world.
        /// </summary>
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        { get { return GetInput(0); } }

        /// <summary>
        /// Reward obtained by one newwork.
        /// Reward is read in every "NumberOfIterations" steps.
        /// </summary>
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Reward
        { get { return GetInput(1); } }


        /// <summary>
        /// Output to the world.
        /// Actions made by the agent.
        /// </summary>
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        /// <summary>
        /// Output size required by the world.
        /// </summary>
        [MyBrowsable, Category("I/O"), YAXSerializableField(DefaultValue = 1)]
        public int OutputSize { get; set; }


        /// <summary>
        /// Number of complexity classes in the outer evolution.
        /// </summary>
        [MyBrowsable, Category("Outer Evolution"), YAXSerializableField(DefaultValue = 10)]
        public int OuterPopulationSize { get; set; }

        /// <summary>
        /// Number of networks created and evaluated for one
        /// complexity class in one iteration of the inner evolution.
        /// </summary>
        [MyBrowsable, Category("Inner Evolution"), YAXSerializableField(DefaultValue = 10)]
        public int InnerPopulationSize { get; set; }

        /// <summary>
        /// Number of steps for the inner evolution.
        /// </summary>
        [MyBrowsable, Category("Inner Evolution"), YAXSerializableField(DefaultValue = 100)]
        public int InnerEvolutionSteps { get; set; }

        /// <summary>
        /// Number of networks created and evaluated for one
        /// complexity class.
        /// </summary>
        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 50)]
        public int MaxCompressionRatio { get; set; }

        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 10)]
        public int MinCompressionRatio { get; set; }

        /// <summary>
        /// Number of networks created and evaluated for one
        /// complexity class.
        /// </summary>
        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 5000)]
        public int MaxNumberOfWeights { get; set; }

        /// <summary>
        /// Number of networks created and evaluated for one
        /// complexity class.
        /// </summary>
        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 500)]
        public int MinNumberOfWeights { get; set; }

        /// <summary>
        /// Maximal number of coefficients.
        /// </summary>
        [MyBrowsable, Category("Network")]
        public int MaxNumberOfCoefficients
        {
            get { return MaxNumberOfWeights / MinCompressionRatio; }
            protected set { Console.WriteLine("set failed"); }
        }

        [MyBrowsable, Category("Network")]
        public int MinNumberOfCoefficients
        {
            get { return MinNumberOfWeights / MaxCompressionRatio; }
            protected set { Console.WriteLine("set failed"); }
        }

        [MyBrowsable, Category("Outer Evolution")]
        public int AllCombinations { get; protected set; }

        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 100)]
        public int WeightSteps { get; set; }

        [MyBrowsable, Category("Network"), YAXSerializableField(DefaultValue = 5)]
        public int CoefficientSteps { get; set; }

        [MyPersistable]
        public MyMemoryBlock<float> PopulationDistribution { get; set; }
        [MyPersistable]
        public MyMemoryBlock<float> PopulationFitnesses { get; set; }

        public MyMemoryBlock<float> NumberOfWeights { get; protected set; }
        public MyMemoryBlock<float> NumberOfCoefficients { get; protected set; }

        //for every sample in inner evolution I get fitness/reward -> utility
        public MyMemoryBlock<float> Fitness { get; protected set; }
        public MyMemoryBlock<float> Utility { get; protected set; }

        //for every sample -> for every coefficient in sample I need mean and sigma (~need covariance matrix)
        [MyPersistable]
        public MyMemoryBlock<float> Means { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> Sigmas { get; protected set; }


        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("Structure")]
        public ActivationFunctionType ACTIVATION_FUNCTION { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int INPUT_UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int HIDDEN_UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int OUTPUT_UNITS { get; protected set; }

        public MyMemoryBlock<float> InputWeights { get; protected set; }
        public MyMemoryBlock<float> RecurrentWeights { get; protected set; }
        public MyMemoryBlock<float> OutputWeights { get; protected set; }

        public MyMemoryBlock<float> InnerEvolWeights { get; protected set; }

        public MyMemoryBlock<float> HiddenActivations { get; protected set; }
        public MyMemoryBlock<float> PreviousHiddenActivations { get; protected set; }
        public MyMemoryBlock<float> OutputActivations { get; protected set; }

        public MyMemoryBlock<float> InnerEvolCoefficients { get; set; }
        public MyMemoryBlock<float> DCTMatrix { get; set; }

        public MyMemoryBlock<float> NotConverged { get; set; }


        [MyBrowsable, Category("Evolution"), YAXSerializableField(DefaultValue = 100)]
        public int NumberOfIterations { get; set; }
        [MyBrowsable, Category("Evolution"), YAXSerializableField(DefaultValue = 5)]
        public int DistributionSmoothness { get; set; }
        [MyBrowsable, Category("Evolution"), YAXSerializableField(DefaultValue = 1f)]
        public float StartSigma { get; set; }

        public int outerSteps;
        public int innerSteps;
        public int sampleSteps;
        public int hiddenLayerSize;
        public int sampleIndex;
        public float threshold;

        public InitTask InitTask { get; protected set; }
        public OuterEvolutionTask OuterEvolutionTask { get; protected set; }
        public InnerEvolutionTask InnerEvolutionTask { get; protected set; }
        public NetworkEvaluation FeedForwardTask { get; protected set; }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                threshold = 0.1f;

                // all possible complexities
                AllCombinations = Math.Max((int)(MaxNumberOfCoefficients - MinNumberOfCoefficients)/CoefficientSteps, 1) 
                    * Math.Max((int)(MaxNumberOfWeights - MinNumberOfWeights)/ WeightSteps, 1);

                OuterPopulationSize = Math.Min(OuterPopulationSize, AllCombinations);

                // distribution and best fitness for complexity class
                PopulationDistribution.Count = AllCombinations;
                PopulationFitnesses.Count = AllCombinations;

                NumberOfWeights.Count = AllCombinations;
                NumberOfCoefficients.Count = AllCombinations;

                Output.Count = OutputSize;

                Fitness.Count = InnerPopulationSize;
                Utility.Count = InnerPopulationSize;

                Means.Count = AllCombinations * MaxNumberOfCoefficients;
                Sigmas.Count = AllCombinations * MaxNumberOfCoefficients;

                // MAX SIZES FOR MAX NETWORK
                INPUT_UNITS = Input.Count;
                OUTPUT_UNITS = Output.Count;
                MinNumberOfWeights = Math.Max(INPUT_UNITS + OUTPUT_UNITS, MinNumberOfWeights);
                MaxNumberOfWeights = Math.Max(MinNumberOfWeights + 1, MaxNumberOfWeights);

                float D = (float)Math.Pow(INPUT_UNITS + OUTPUT_UNITS, 2) + 4 * MaxNumberOfWeights;
                HIDDEN_UNITS = Math.Max(1, (-(INPUT_UNITS + OUTPUT_UNITS) + (int)Math.Sqrt(D)) / 2);

                InputWeights.Count = HIDDEN_UNITS * INPUT_UNITS;
                RecurrentWeights.Count = HIDDEN_UNITS * HIDDEN_UNITS;
                OutputWeights.Count = HIDDEN_UNITS * OUTPUT_UNITS;

                HiddenActivations.Count = HIDDEN_UNITS;
                PreviousHiddenActivations.Count = HIDDEN_UNITS;
                OutputActivations.Count = OUTPUT_UNITS;

                InnerEvolWeights.Count = MaxNumberOfWeights;
                InnerEvolCoefficients.Count = MaxNumberOfCoefficients;
                DCTMatrix.Count = MaxNumberOfCoefficients * MaxNumberOfWeights;

                NotConverged.Count = AllCombinations;
            }
            //sizes all set
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Input.Count != 0, this, "Zero input size is not allowed.");

            base.Validate(validator);
            validator.AssertError(Reward.Count != 0, this, "Zero reward size is not allowed.");
        }

    }

}
