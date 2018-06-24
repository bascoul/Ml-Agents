using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class TennisAgent3D : Agent
{
    [Header("Specific to Tennis")]
    public GameObject ball;
    public bool invertX;
    public int score;
    public GameObject scoreText;
    public GameObject myArea;
    public GameObject opponent;

    private Text textComponent;
    private Rigidbody agentRb;
    private Rigidbody ballRb;
    private float invertMult;

    public override void InitializeAgent()
    {
        agentRb = GetComponent<Rigidbody>();
        ballRb = GetComponent<Rigidbody>();
        textComponent = scoreText.GetComponent<Text>();
    }

    public override void CollectObservations()
    {
        AddVectorObs(invertMult * (transform.position.x - myArea.transform.position.x));
        AddVectorObs(transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * (transform.position.z - myArea.transform.position.z));
        AddVectorObs(invertMult * agentRb.velocity.x);
        AddVectorObs(agentRb.velocity.y);
        AddVectorObs(invertMult * agentRb.velocity.z);

        AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * (ball.transform.position.z - myArea.transform.position.z));
        AddVectorObs(invertMult * ballRb.velocity.x);
        AddVectorObs(invertMult * ballRb.velocity.z);
        AddVectorObs(ballRb.velocity.y);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f) * invertMult;
        var moveZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * invertMult;
        
        agentRb.velocity = new Vector3(moveX * 30f, 0f, moveZ * 30f);

        AddReward(0.01f);
        textComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        invertMult = invertX ? -1f : 1f;

        transform.position = new Vector3(-invertMult * Random.Range(6f, 8f), -2.5f, -invertMult * Random.Range(-3f, 3f)) + transform.parent.transform.position;
        agentRb.velocity = new Vector3(0f, 0f, 0f);
    }
}
