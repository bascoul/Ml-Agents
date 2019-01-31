using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;

#endif


public class FPSCounter : MonoBehaviour
{

	
	
	public float experimentTime;
	public GameObject Cralwer;
	public List<int> NAgents;
	private int currentIndex;
	private int currentNumberAgent;
	private int nFrames;
	private string result = "";
	
	
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update ()
	{
//		timeDelay += Time.deltaTime;
		if (currentIndex > NAgents.Count)
		{
			return;
		}

		if (Time.realtimeSinceStartup > (currentIndex+1) * experimentTime)
		{
			result += currentNumberAgent + " Crawlers : " +
			          nFrames / experimentTime + " FPS\n";
			nFrames = 0;

			if (currentIndex == NAgents.Count)
			{
				Debug.Log(result);
				var logPath = Path.GetFullPath(".") + "/TestResults.txt";
				var logWriter = new StreamWriter(logPath, false);
				logWriter.WriteLine(System.DateTime.Now.ToString());
				logWriter.WriteLine(result);
				logWriter.Close();
			
				Application.Quit();
				currentIndex++;
				return;
			}
			
			var n = NAgents[currentIndex];
			currentIndex++;
			
			for (var nn = 0; nn < n; nn++)
			{
				GameObject.Instantiate(Cralwer,
					new Vector3(
						0,
						0,
						(currentNumberAgent+nn) * 20),
					default(Quaternion));
			}
			currentNumberAgent += n;
		}
		nFrames += 1;
	}
}
