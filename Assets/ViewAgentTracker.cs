using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ViewAgentTracker : MonoBehaviour
{

    public GameObject player;
    //private Vector3 offset;
    //private Vector3 rotation;

    // Start is called before the first frame update
    void Start()
    {
        //offset = this.transform.position - player.transform.position;
    }

    // Update is called once per frame
    void LateUpdate()
    {
        this.transform.position = player.transform.position;// + offset;
        this.transform.rotation = Quaternion.Euler(player.transform.rotation.eulerAngles.y * Vector3.up);
    }
}
