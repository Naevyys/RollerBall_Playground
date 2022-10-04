using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ViewAgentTracker : MonoBehaviour
{

    public GameObject player;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void LateUpdate()
    {
        this.transform.position = player.transform.position;
        this.transform.rotation = Quaternion.Euler(player.transform.rotation.eulerAngles.y * Vector3.up);
    }
}
