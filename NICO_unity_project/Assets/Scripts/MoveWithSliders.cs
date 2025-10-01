using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class MoveWithSliders : MonoBehaviour
{
    ArticulationBody nico_art_body;

    [Header("Collected Joints")]
    public List<ArticulationBody> leftHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> rightHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> headJoints = new List<ArticulationBody>();

    public List<ArticulationBody> activeJoints;
    public List<float> jointTargets = new List<float>();

    private Dictionary<ArticulationBody, ArticulationBody[]> fingerChains = new Dictionary<ArticulationBody, ArticulationBody[]>();

    private void CacheFingerChain(ArticulationBody root)
    {
        if (root == null) return;
        var joints = root.GetComponentsInChildren<ArticulationBody>();
        fingerChains[root] = joints.Where(j => j != root).ToArray();
    }

    private void CollectJointsInformation(ArticulationBody root)
    {
        foreach (var joint in root.GetComponentsInChildren<ArticulationBody>())
        {
            string joint_name = joint.name.ToLower();
            string joint_tag = joint.tag.ToLower();

            // caching the basis of the fingers to be later used for movements of all joints in the finger with one parameter
            if (joint_tag.Contains("base"))
            {
                CacheFingerChain(joint);
            }

            if (joint_name.StartsWith("l_") && !joint_tag.Contains("continuation"))
                leftHandJoints.Add(joint);

            else if (joint_name.StartsWith("r_") && !joint_tag.Contains("continuation"))
                rightHandJoints.Add(joint);

            else if (joint_name.StartsWith("h_"))
                headJoints.Add(joint);
        }
    }

    private void UpdateActiveJoints()
    {
        activeJoints = new List<ArticulationBody>();
        activeJoints.AddRange(leftHandJoints);
        activeJoints.AddRange(rightHandJoints);
        activeJoints.AddRange(headJoints);
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        nico_art_body = GetComponent<ArticulationBody>();

        CollectJointsInformation(nico_art_body);
        UpdateActiveJoints();
    }

    public void SetJointTarget(ArticulationBody joint, float angle)
    {
        var drive = joint.xDrive;
        drive.target = angle;
        joint.xDrive = drive;

        if (fingerChains.TryGetValue(joint, out var chain))
        {
            foreach (var sub_joint in chain)
            {
                float ratio = 1f;
                if (sub_joint.tag.ToLower().Contains("thumb"))
                {
                    ratio = 0.25f;
                }

                SetJointTarget(sub_joint, angle * ratio);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        int i = 0;
        if (jointTargets.Count > 0 )
        {
            foreach (var joint in activeJoints)
            {
                // Debug.Log(i);
                SetJointTarget(joint, jointTargets[i]);
                i++;
            }
        }
        
    }
}
