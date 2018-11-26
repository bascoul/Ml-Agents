using System.Collections.Generic;
using System.Linq;
using System;
using System.Runtime.InteropServices;
using UnityEngine.Profiling;
using Barracuda;  

namespace UnityEngine.MachineLearning.InferenceEngine
{

    public class BarracudaModelMetadata : ModelMetadata
    {
        public IEnumerable<Tensor> InputFeatures()
        {
            return m_Engine.InputFeatures();
        }

        public IEnumerable<Tensor> OutputFeatures()
        {
            return m_Engine.OutputFeatures();
        }

        public int GetIntConstant(string name)
        {
            return (int)m_Engine.GetModel().GetTensorByName(name)[0];
        }

        public float GetFloatConstant(string name)
        {
            return (float)m_Engine.GetModel().GetTensorByName(name)[0];
        }

        public Tensor GetTensorConstant(string name)
        {
            return m_Engine.FromBarracuda(m_Engine.GetModel().GetTensorByName(name));
        }

        public string[] GetInputFeatureNames()
        {
            if (m_InputNames == null)
                InitInputNames();

            return m_InputNames;
        }

        public string[] GetOutputFeatureNames()
        {
            if (m_OutputNames == null)
                InitOutputNames();

            return m_OutputNames;
        }

        public long[] GetTensorShape(string name)
        {
            return m_Engine.FromBarracuda(m_Engine.GetModel().GetShapeByName(name));
        }

        public BarracudaModelMetadata(BarracudaInferenceEngine engine)
        {
            m_Engine = engine;
        }

        void InitInputNames()
        {
            List<string> names = new List<string>();
            foreach (var input in m_Engine.GetModel().inputs)
            {
                names.Add(input.name);
            }

            m_InputNames = names.ToArray();
        }
        
        void InitOutputNames()
        {
            List<string> names = new List<string>();
            foreach (var output in m_Engine.GetModel().outputs)
            {
                names.Add(output);
            }

            m_OutputNames = names.ToArray();
        }

        private BarracudaInferenceEngine m_Engine;
        private string[] m_InputNames;
        private string[] m_OutputNames;
    }
    
    /// <summary>
    /// BarracudaInferenceEngine - Inference engine utilizing the Barracuda library for cross-platform inference
    /// </summary>
    [InferenceEngine(ModelFormat.Barracuda)]
    public class BarracudaInferenceEngine : InferenceEngine, IDisposable
    {
        private Barracuda.Model m_Model;
        private Barracuda.IWorker m_Worker;
        private BarracudaModelMetadata m_ModelMetadata;

        // TODO: InferenceEngine implements IDisposable
        public void Dispose()
        {
            m_Worker?.Dispose();
        }

        internal Barracuda.Model GetModel()
        {
            return m_Model;
        }

        public void PrepareModel(Model model, InferenceEngineConfig config)
        {
            Profiler.BeginSample("BarracudaInferenceEngine.PrepareModel");

            if (model.ModelFormat != ModelFormat.Barracuda)
                throw new ArgumentException("Supplied model is not in Barracuda format (" + model.ModelFormat + ")");

            m_Model = Barracuda.ModelLoader.Load(model.ModelData, config.Verbose);
            m_Worker = Barracuda.BarracudaWorkerFactory.CreateWorker(GetWorkerType(config), m_Model, config.Verbose);
            
            Profiler.EndSample();
        }

        public int ExecuteGraph(IEnumerable<Tensor> inputs_it, IEnumerable<Tensor> outputs_it)
        {
            Profiler.BeginSample("BarracudaInferenceEngine.ExecuteGraph");
            var inputs = inputs_it.ToArray();
            var outputs = outputs_it.ToArray();

            foreach (var i in inputs)
                m_Worker.AddInput(i.Name, ToBarracuda(i));
            
            m_Worker.Execute();

            for (int q = 0; q < outputs.Length; ++q)
            {
                var t = m_Worker.Fetch(outputs[q].Name);
                CopyDataFromBarracudaAndDispose(t, ref outputs[q]);
            }

            Profiler.EndSample();
            return 0;
        }

        public ModelMetadata GetModelMetadata()
        {
            if (m_ModelMetadata == null)
                m_ModelMetadata = new BarracudaModelMetadata(this);

            return m_ModelMetadata;
        }

        public IEnumerable<Tensor> InputFeatures()
        {
            List<Tensor> tensors = new List<Tensor>();
            foreach (var input in m_Model.inputs)
            {
                tensors.Add(new Tensor
                {
                    Name = input.name,
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null,
                    Shape = input.shape.Select(i => (long)i).ToArray()
                });
            }

            return tensors;
        }

        public IEnumerable<Tensor> OutputFeatures()
        {
            List<Tensor> tensors = new List<Tensor>();
            foreach (var tensorName in m_Model.outputs)
            {
                tensors.Add(new Tensor
                {
                    Name = tensorName,
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null,
                    Shape = FromBarracuda(m_Model.GetShapeByName(tensorName))
                });
            }

            return tensors;
        }

        public bool AllocateOutputs()
        {
            return true;
        }

        // ----------------------------------------------------------
        internal virtual Barracuda.BarracudaWorkerFactory.Type GetWorkerType(InferenceEngineConfig config)
        {
            switch (config.Device)
            {
                case InferenceEngineConfig.DeviceType.GPU:
                    return Barracuda.BarracudaWorkerFactory.Type.ComputeFast;

                case InferenceEngineConfig.DeviceType.CPU:
                default:
                    return Barracuda.BarracudaWorkerFactory.Type.CSharpFast;

            }
        }

        private Array LinearizeArray(Array src)
        {
            var elementType = src.GetType().GetElementType();
            var elementSize = Marshal.SizeOf(elementType);
            var dest = Array.CreateInstance(elementType, src.Length);
            Buffer.BlockCopy(src, 0, dest, 0, src.Length * elementSize);
            return dest;
        }

        private Array ReshapeArray(Array src, long[] shape)
        {
            var elementType = src.GetType().GetElementType();
            var elementSize = Marshal.SizeOf(elementType);
            var dest = Array.CreateInstance(elementType, shape);
            Buffer.BlockCopy(src, 0, dest, 0, src.Length * elementSize);
            return dest;
        }

        protected Barracuda.TensorShape ToBarracuda(long[] src)
        {
            if (src.Length > 4)
                throw new NotImplementedException("Barracuda does not support Tensor shapes with rank higher than 4");

            var shape = new int[4];
            for (var axis = 0; axis < src.Length; ++axis)
                shape[shape.Length-axis-1] = (int)src[src.Length-axis-1];

            return new Barracuda.TensorShape(shape);
        }

        internal long[] FromBarracuda(Barracuda.TensorShape src)
        {
            return src.ToArray().Select(i => (long)i).ToArray();
        }

        internal virtual Barracuda.Tensor ToBarracuda(Tensor src)
        {
            Profiler.BeginSample("BarracudaInferenceEngine.ToBarracuda");

            if (src.ValueType != Tensor.TensorType.FloatingPoint)
                throw new NotImplementedException("Barracuda does not support non-float Tensors");

            var shape = ToBarracuda(src.Shape);
            return new Barracuda.Tensor(shape, LinearizeArray(src.Data) as float[], src.Name);
        }

        internal virtual Tensor FromBarracuda(Barracuda.Tensor src)
        {
            Profiler.BeginSample("BarracudaInferenceEngine.FromBarracuda");

            var shape = FromBarracuda(src.shape);
            return new Tensor
            {
                Name = src.name,
                ValueType = Tensor.TensorType.FloatingPoint,
                Shape = shape,
                Data = ReshapeArray(src.data.Download(src.length), shape)
            };
        }

        internal virtual void CopyDataFromBarracudaAndDispose(Barracuda.Tensor src, ref Tensor dst)
        {
            Profiler.BeginSample("BarracudaInferenceEngine.CopyDataFromBarracudaAndDispose");

            long dstLength = 1;
            foreach (var dimLength in dst.Shape)
                dstLength *= dimLength;

            if (src.length != dstLength)
                throw new NotImplementedException("Source and destination Tensor sizes do not match");

            dst.Data = ReshapeArray(src.data.Download(src.length), dst.Shape);
            src.Dispose();
        }

    }
}
