using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;
using System;
using Random = UnityEngine.Random;


public class NicoAgent : Agent
{
    ArticulationBody nico_art_body;

    [Header("NICO parts to control")]
    [SerializeField] private bool useLeftHand = true;
    [SerializeField] private bool useRightHand = true;
    [SerializeField] private bool useHead = true;

    [Header("Collected Joints")]
    public List<ArticulationBody> leftHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> rightHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> headJoints = new List<ArticulationBody>();

    private List<ArticulationBody> activeJoints;

    private Dictionary<ArticulationBody, float> defaultJointTargets = new Dictionary<ArticulationBody, float>();
    private Dictionary<ArticulationBody, float> currentJointChanges = new Dictionary<ArticulationBody, float>();

    private Dictionary<ArticulationBody, ArticulationBody[]> fingerChains = new Dictionary<ArticulationBody, ArticulationBody[]>();

    [Tooltip("The target object")]
    public GameObject target;

    [Tooltip("End effector")]
    public GameObject effector_1;

    [Tooltip("End effector")]
    public GameObject effector_2;

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

        if (useLeftHand && leftHandJoints.Count > 0)
            activeJoints.AddRange(leftHandJoints);

        if (useRightHand && rightHandJoints.Count > 0)
            activeJoints.AddRange(rightHandJoints);

        if (useHead && headJoints.Count > 0)
            activeJoints.AddRange(headJoints);
    }


    public override void Initialize()
    {
        base.Initialize();

        nico_art_body = GetComponent<ArticulationBody>();

        CollectJointsInformation(nico_art_body);
        UpdateActiveJoints();

        foreach (var joint in activeJoints)
        {
            defaultJointTargets[joint] = joint.xDrive.target;
            currentJointChanges[joint] = 0f;
        }
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


    public void ResetJoint(ArticulationBody joint, float joint_default)
    {
        // Reset drive target
        var drive = joint.xDrive;
        drive.target = joint_default;
        joint.xDrive = drive;

        // Reset physics state
        joint.jointPosition = new ArticulationReducedSpace(joint_default); 
        joint.jointVelocity = new ArticulationReducedSpace(0.0f);

        // Reset rigidbody motion
        joint.linearVelocity = Vector3.zero;
        joint.angularVelocity = Vector3.zero;

        if (fingerChains.TryGetValue(joint, out var chain))
        {
            foreach (var sub_joint in chain)
            {
                ResetJoint(sub_joint, joint_default);
            }
        }
    }


    public override void OnEpisodeBegin()
    {
        // move target cube to a random position
        target.transform.position = new Vector3(Random.Range(0.2f, 1.25f), Random.Range(1.35f, 1.8f), Random.Range(-0.2f, 0.5f));
        target.transform.localRotation = Quaternion.Euler(0f, 0f, 0f);

        // reset joint positions, velocities, targets
        foreach (var joint in activeJoints)
        {
            ResetJoint(joint, defaultJointTargets[joint]);
            currentJointChanges[joint] = 0f;
        }
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // joint target values are used for observations.
        foreach (var joint in activeJoints)
        {   
            float joint_target = joint.xDrive.target;

            float lower_limit = joint.xDrive.lowerLimit;
            float upper_limit = joint.xDrive.upperLimit;

            // each joint is normalized within its allowed range, the minimum gets mapped to 0 and maximum to 1 regardless the values.
            float normalized_joint_target = (joint_target - lower_limit) / (upper_limit - lower_limit);

            sensor.AddObservation(normalized_joint_target);
        }
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        float max_change = 2f;          // maximal allowed change (in degrees) per step in the DecisionRequester
        float change_magnitude = 0.3f;  // maximal allowed change of the change per action

        foreach (var (joint, action) in activeJoints.Zip(actions.ContinuousActions, (x, y) => (x, y)))
        {
            currentJointChanges[joint] = Mathf.Clamp(currentJointChanges[joint] + action * change_magnitude, -max_change, max_change);
            float new_target = Mathf.Clamp(joint.xDrive.target + currentJointChanges[joint], joint.xDrive.lowerLimit, joint.xDrive.upperLimit);
            SetJointTarget(joint, new_target);
        }

        float right_reward = effector_1.transform.position.y;   // no need for scaling. In this task max pos.y is around 1
        float left_reward = effector_2.transform.position.y;    // no need for scaling. In this task max pos.y is around 1

        SetReward((right_reward + left_reward) / 2);  // normalize to keep rewards for individual steps below 1
    }
}
