using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MLAgents;
using UnityEditor;
using UnityEngine;

public class FixedUpdateTest : MonoBehaviour
{
    public bool useCaptureFrameRate = false;
    public int captureFrameRate = 60;
    public int frameRate =60;
    public float experienceDuration=5;
    public int sleepUpdate=50;
    public int sleepFixedUpdate=5;
    
    [Range(1, 100)] public float timeScale=1;

    private Clock clock = new Clock();
    private bool record;
    private int n_update;
    private int n_fixedUpdate;
    

    // Use this for initialization
    void Start () {
        if (useCaptureFrameRate)
        {
            Time.captureFramerate = captureFrameRate;
        }
        Application.targetFrameRate = frameRate;
        Time.timeScale = timeScale;

    }

    private void ExperiemntLoop()
    {
        // This time is for the experiment to start some time after the begening of the game
        // in order to avoid the overhead of the start of the engine
        float startTime = 5;
        if ((Time.realtimeSinceStartup > startTime) && !record)
        {
            record = true;
            clock.Reset();
        }

        if ((Time.realtimeSinceStartup > clock.realZero + experienceDuration) && record)
        {
            record = false;
            Debug.Log(GetConfig()+clock.GetTime() + GetUpdates());
            UnityEditor.EditorApplication.isPlaying = false;
        }
    }
    
    private void FixedUpdate()
    {
        
        System.Threading.Thread.Sleep(sleepFixedUpdate);
        ExperiemntLoop();

        if (record)
        {
            n_fixedUpdate++;
        }
    }

    // Update is called once per frame
    void Update () 
    {
        System.Threading.Thread.Sleep(sleepUpdate);
        if (record)
        {
            n_update++;
        }
    }

    private string GetConfig()
    {
        return "use CaptureFrameRate : " + useCaptureFrameRate +
               "\nCaptureFrameRate : " + captureFrameRate +
               "\nframe rate : " + Application.targetFrameRate +
               "\nexperience duration real seconds : " + experienceDuration +
               "\nTimeScale : " + timeScale+
               "\nSleep Update : " + sleepUpdate+
               "\nSleep Fixed Update : " + sleepFixedUpdate;
    }
    
    private string GetUpdates()
    {
        return "\n\nnumber updates : " + n_update
                                     + "\nnumber fixed updates : " + n_fixedUpdate;
    }

    
    private class Clock
    {
        public float fakeZero;
        public float realZero;

        public void Reset()
        {
            fakeZero = Time.time;
            realZero = Time.realtimeSinceStartup;
        }

        public string GetTime()
        {
            return "\n\nGameTime : " + (Time.time - fakeZero) 
                   +" \nReal Time : " + (Time.realtimeSinceStartup - realZero)
                +"\nScaled Real Time : "+(Time.realtimeSinceStartup - realZero)*Time.timeScale;
        }
    }
    
    
}
