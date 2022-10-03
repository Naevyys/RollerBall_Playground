using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    void Start () {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void AgentReset()
    {
        if (this.transform.position.y < 0)
        {
            // If the Agent fell, zero its momentum
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.position = new Vector3( 0, 0.5f, 0);
            // Rotate agent randomly
            this.transform.rotation = Quaternion.Euler(Vector3.up * Random.Range(0f, 360f));
        }

        // Move the target to a new spot
        Target.position = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }

    public override void CollectObservations()
    {
        // Target and Agent positions
        // AddVectorObs(Target.position);  // We don't want the agent to know the target position anymore
        // AddVectorObs(this.transform.position);

        // Agent rotation
        // AddVectorObs(this.transform.rotation);

        // Agent velocity
        AddVectorObs(rBody.velocity.x);
        AddVectorObs(rBody.velocity.z);
    }

    public float speed = 10;
    public override void AgentAction(float[] vectorAction)
    {
        // Actions, size = 2 (first is for movement, second for rotation)

        // Move
        Vector3 controlSignalMovement = this.transform.forward * vectorAction[0]; // Move forward only
        rBody.AddForce(controlSignalMovement * speed);

        // Turn
        Vector3 controlSignalRotation = this.transform.up * vectorAction[1];
        rBody.AddTorque(controlSignalRotation);

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

    }

    public override float[] Heuristic()
    {
        var action = new float[2];
        action[0] = Input.GetAxis("Horizontal");
        action[1] = Input.GetAxis("Vertical");
        return action;
    }

}
