using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TestLoader : MonoBehaviour
{
	public float captureFPS = 3f;
	public int captureFrames = 15;
	public string[] testScenes;

	// Use this for initialization
	void Start () 
	{
		DontDestroyOnLoad(gameObject);

		StartCoroutine(DoTesting());
	}

	string GetScreenShotDir()
	{
		if (Application.isMobilePlatform)
			return Application.persistentDataPath;

		return Directory.GetCurrentDirectory();
	}
	
	// Update is called once per frame
	IEnumerator DoTesting ()
	{
		//Time.captureFramerate = 60;
		
		foreach (var sceneName in testScenes)
		{
			SceneManager.LoadScene(sceneName);
			yield return null;
			Debug.Log("Saving screenshots to.. " + GetScreenShotDir());

			for (int frame = 0; frame < captureFrames; frame++)
			{
				ScreenCapture.CaptureScreenshot(string.Format("{0}_{1:00}.png", sceneName, frame));
				yield return new WaitForSeconds(1f/captureFPS);			
			}
			
			yield return null;
		}

		yield return null;

		Time.captureFramerate = 0;
		
		Application.Quit();
	}
}
