using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    public Transform Target;
    private float speed = 1f;
    private float rotationScalingFactor = 90f;  //f means float

    void Start () {
        rBody = GetComponent<Rigidbody>();
    }

    public override void AgentReset()
    {
        // Zero momentum
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;

        // Move the target to a new spot
        Target.position = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        // Move agent back to center
        this.transform.position = new Vector3( 0, 0.5f, 0);
        // Rotate agent randomly
        this.transform.rotation = Quaternion.Euler(Vector3.up * Random.Range(0f, 360f));
    }

    public override void CollectObservations()
    {

    }

    public override void AgentAction(float[] vectorAction)
    {
        // Actions, size = 2 (first is for movement, second for rotation)

        // Move
        Vector3 controlSignalMovement = (Vector3.forward * vectorAction[0]);
        rBody.transform.Translate(controlSignalMovement * speed * Time.deltaTime);  // Translate is relative to Space.Self by default, so need to use Vector3.forward...

        // Turn
        Vector3 controlSignalRotation = Vector3.up * vectorAction[1];
        rBody.transform.Rotate(controlSignalRotation * rotationScalingFactor * Time.deltaTime);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.position, Target.position);

        // Reached target
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            Done();
        }

        // Fell off platform
        if (this.transform.position.y < 0)
        {
            SetReward(-1.0f);  // Negative reward  if it falls down the platform
            Done();
        }

        // If not done, add small negative reward to encourage the agent to solve the task as fast as possible
        SetReward(-0.01f);
    }

    public override float[] Heuristic()
    {
        var action = new float[2];
        action[1] = Input.GetAxis("Horizontal");
        action[0] = Input.GetAxis("Vertical");
        return action;
    }

}
