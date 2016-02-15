using GoodAI.Core;
using GoodAI.Core.Task;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EvolutionModule.Tasks
{
    /// <summary>
    /// Task for calculating output of the network.
    /// </summary>
    [Description("Feed forward evaluation")]
    public class NetworkEvaluation : MyTask<EvolutionNode>
    {

        private MyCudaKernel m_feedForwardHiddenKernel; 
        private MyCudaKernel m_feedForwardOutputKernel;

        /// <summary>
        /// Initializing kernels for feedforward computation.
        /// </summary>
        public override void Init(int nGPU)
        {
            m_feedForwardHiddenKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\ForwardPassKernel", "ForwardPassHiddenKernel");
            m_feedForwardHiddenKernel.SetupExecution(Owner.HIDDEN_UNITS);
            m_feedForwardHiddenKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_feedForwardHiddenKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

            m_feedForwardHiddenKernel.SetConstantVariable("D_ACTIVATION_FUNCTION", (int)Owner.ACTIVATION_FUNCTION);
            m_feedForwardHiddenKernel.DynamicSharedMemory = sizeof(float) * (uint)(Owner.INPUT_UNITS + Owner.HIDDEN_UNITS);

            m_feedForwardOutputKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\ForwardPassKernel", "ForwardPassOutputKernel");
            m_feedForwardOutputKernel.SetupExecution(Owner.OUTPUT_UNITS);
            m_feedForwardOutputKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_feedForwardOutputKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_feedForwardOutputKernel.SetConstantVariable("D_ACTIVATION_FUNCTION", (int)Owner.ACTIVATION_FUNCTION);
            m_feedForwardOutputKernel.DynamicSharedMemory = sizeof(float) * (uint)Owner.HIDDEN_UNITS;
        }

        /// <summary>
        /// Method executes kernels for calculating the output of the
        /// evaluated network. OutputActivations calculated in the kernels
        /// is then copied to the Outpu from the EvolutionNode.</summary>
        public override void Execute()
        {
            Owner.HiddenActivations.CopyToMemoryBlock(Owner.PreviousHiddenActivations, 0, 0, Owner.hiddenLayerSize);

            //compute activation of hidden layer
            m_feedForwardHiddenKernel.Run(
                 Owner.Input,
                 Owner.HiddenActivations,
                 Owner.PreviousHiddenActivations,
                 Owner.InputWeights,
                 Owner.RecurrentWeights,
                 Owner.hiddenLayerSize
                );

            //compute activation of output layer
            m_feedForwardOutputKernel.Run(
                Owner.HiddenActivations,
                Owner.OutputActivations,
                Owner.OutputWeights,
                Owner.hiddenLayerSize
                );

            Owner.OutputActivations.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.OUTPUT_UNITS);
        }
    }
}
