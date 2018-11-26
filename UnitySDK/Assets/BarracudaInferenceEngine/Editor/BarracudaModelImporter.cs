using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;
using UnityEngine.MachineLearning.InferenceEngine;

namespace UnityEditor.MachineLearning.InferenceEngine
{
	/// <summary>
	/// BarracudaModelImporter - An implementation of ScriptedImporter that loads an 
	/// Inference API Model asset into Unity.
	/// </summary>
	[ScriptedImporter(1, "barracuda" )]
	public class BarracudaModelImporter : ScriptedImporter
	{
		public override void OnImportAsset(AssetImportContext ctx)
		{
			Debug.Log("Importing model " + Path.GetFileName(ctx.assetPath));

			var inputType = Path.GetExtension(ctx.assetPath);
			if (inputType == null)
			{
				throw new Exception("Model doesn't have an extension! This is probably a bug in the ModelImporter.");
			}

			var model = ScriptableObject.CreateInstance<Model>();

			model.ModelData = File.ReadAllBytes(ctx.assetPath);
			model.ModelFormat = ModelFormat.Barracuda;
			Debug.Log("Format detected: " + model.ModelFormat.ToString());

			userData = model.ModelFormat.ToString();

			// TODO: icon not showing in the component
			Texture2D texture =
				(Texture2D) AssetDatabase.LoadAssetAtPath("Assets/BarracudaInferenceEngine/Resources/Barracuda.png",
					typeof(Texture2D));

#if UNITY_2017_3_OR_NEWER
            ctx.AddObjectToAsset(ctx.assetPath, model, texture);
            ctx.SetMainObject(model);
#else
			ctx.SetMainAsset(ctx.assetPath, model, texture);
#endif
		} 
	}

}
