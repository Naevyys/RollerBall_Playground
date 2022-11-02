using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    private Vector3 initial_position;
    private Quaternion initial_rotation;
    private Vector3 target_initial_position;
    public Transform Target;
    private float speed = 5f;
    private float rotationScalingFactor = 90f;  //f means float

    void Start () {
        rBody = GetComponent<Rigidbody>();
        initial_position = this.transform.position;
        initial_rotation = this.transform.rotation;
        target_initial_position = Target.transform.position;
    }

    public override void AgentReset()
    {
        // Zero momentum
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;

        Target.position = target_initial_position;  // Reset target tp initial position
        this.transform.position = initial_position;  // Reset agent to initial position
        this.transform.rotation = initial_rotation;  // Reset agent to initial rotation
    }

    public override void CollectObservations()
    {

    }

    public override void AgentAction(float[] vectorAction)
    {
        // Actions, size = 2 (first is for movement, second for rotation)

        // Move
        Vector3 controlSignalMovement = (Vector3.forward * (vectorAction[0] + 1) / 2);
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
