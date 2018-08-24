using UnityEngine;
using UnityEngine.UI;
using MLAgents;

public class TennisAgent : Agent
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
    public bool use3D;

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
        AddVectorObs(invertMult * agentRb.velocity.x);
        AddVectorObs(agentRb.velocity.y);

        AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
        AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
        AddVectorObs(invertMult * ballRb.velocity.x);
        AddVectorObs(ballRb.velocity.y);

        if (use3D)
        {
            AddVectorObs(invertMult * (transform.position.z - myArea.transform.position.z));
            AddVectorObs(invertMult * agentRb.velocity.z);
            AddVectorObs(invertMult * (ball.transform.position.z - myArea.transform.position.z));
            AddVectorObs(invertMult * ballRb.velocity.z);

            AddVectorObs(transform.rotation.eulerAngles.y);
            AddVectorObs(transform.rotation.eulerAngles.z * invertMult);
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var moveX = Mathf.Clamp(vectorAction[0], -1f, 1f);

        if (use3D)
        {            
            var moveZ = Mathf.Clamp(vectorAction[1], -1f, 1f);
            var angleY = Mathf.Clamp(vectorAction[2], -1f, 1f);
            var angleZ = Mathf.Clamp(vectorAction[3], -1f, 1f);
            
            agentRb.velocity = new Vector3(moveX * 20f * invertMult, 0f, moveZ * 20f * invertMult);
            transform.rotation = Quaternion.Euler(0f, angleY * 20f, angleZ * 20f + 60f * invertMult);
        }
        else
        {
            var moveY = Mathf.Clamp(vectorAction[1], -1f, 1f);            
            if (moveY > 0.5 && transform.position.y - transform.parent.transform.position.y < -1.5f)
            {
                agentRb.velocity = new Vector3(agentRb.velocity.x, 7f, 0f);
            }

            agentRb.velocity = new Vector3(moveX * 30f, agentRb.velocity.y, 0f);

            if (invertX && transform.position.x - transform.parent.transform.position.x < -invertMult || 
                !invertX && transform.position.x - transform.parent.transform.position.x > -invertMult)
            {
                transform.position = new Vector3(-invertMult + transform.parent.transform.position.x, 
                    transform.position.y, 
                    transform.position.z);
            }

        }
        
        AddReward(0.01f);
        textComponent.text = score.ToString();
    }

    public override void AgentReset()
    {
        invertMult = invertX ? -1f : 1f;

        var zOffset = use3D ? -invertMult * Random.Range(-3f, 3f) : 0f;
        transform.position = new Vector3(-invertMult * Random.Range(6f, 8f), -2.5f, zOffset) +
                             transform.parent.transform.position;
        agentRb.velocity = new Vector3(0f, 0f, 0f);
    }
}
