using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;
using System;
using Random = UnityEngine.Random;


public class NicoAgentReach : Agent
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

    // added START
    [Header("Reach Reward Settings")]
    [SerializeField] private float reachDistance = 0.05f;
    [SerializeField] private float progressWeight = 3f;
    [SerializeField] private float distancePenaltyWeight = 0.02f;
    [SerializeField] private float reachBonus = 1f;
    [SerializeField] private float smoothnessWeight = 0.01f;
    [SerializeField] private float movementWeight = 0.005f;

    [Header("Stall Detection")]
    [SerializeField] private int noProgressStepsToLock = 100;
    [SerializeField] private float noProgressEpsilon = 0.001f;

    private bool rightHandLocked;
    private bool leftHandLocked;

    private float previousRightDistance;
    private float previousLeftDistance;

    private float bestRightDistance;
    private float bestLeftDistance;

    private int rightNoProgressSteps;
    private int leftNoProgressSteps;
    // added END


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

        rightHandLocked = false;
        leftHandLocked = false;

        rightNoProgressSteps = 0;
        leftNoProgressSteps = 0;

        previousRightDistance = Vector3.Distance(
            effector_1.transform.position, target.transform.position);

        previousLeftDistance = Vector3.Distance(
            effector_2.transform.position, target.transform.position);

        bestRightDistance = previousRightDistance;
        bestLeftDistance = previousLeftDistance;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (var joint in activeJoints)
        {
            float lower = joint.xDrive.lowerLimit;
            float upper = joint.xDrive.upperLimit;
            float targetAngle = joint.xDrive.target;

            float range = upper - lower;

            float normalizedTarget = Mathf.Abs(range) < 0.0001f
                ? 0f
                : Mathf.Clamp01((targetAngle - lower) / range);

            sensor.AddObservation(normalizedTarget);
        }

        // Relative cube positions give the agent information about
        // where each hand must move.
        sensor.AddObservation(
            target.transform.position - effector_1.transform.position);

        sensor.AddObservation(
            target.transform.position - effector_2.transform.position);

        // Inform the policy when an arm has been locked.
        sensor.AddObservation(rightHandLocked ? 1f : 0f);
        sensor.AddObservation(leftHandLocked ? 1f : 0f);
    }


    private bool IsJointLocked(ArticulationBody joint)
    {
        if (rightHandLocked && rightHandJoints.Contains(joint))
            return true;

        if (leftHandLocked && leftHandJoints.Contains(joint))
            return true;

        return false;
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        float max_change = 2f;
        float change_magnitude = 0.3f;

        float smoothnessSum = 0f;
        float movementSum = 0f;
        int controlledJointCount = 0;

        // Apply actions only to unlocked joints.
        foreach (var (joint, action) in activeJoints.Zip(
            actions.ContinuousActions, (x, y) => (x, y)))
        {
            if (IsJointLocked(joint))
            {
                currentJointChanges[joint] = 0f;
                continue;
            }

            float previousChange = currentJointChanges[joint];

            currentJointChanges[joint] = Mathf.Clamp(
                previousChange + action * change_magnitude,
                -max_change,
                max_change);

            float newTarget = Mathf.Clamp(
                joint.xDrive.target + currentJointChanges[joint],
                joint.xDrive.lowerLimit,
                joint.xDrive.upperLimit);

            SetJointTarget(joint, newTarget);

            // Penalize abrupt changes in movement.
            smoothnessSum += Mathf.Abs(
                currentJointChanges[joint] - previousChange) / max_change;

            // Small penalty for continuous joint movement.
            movementSum += Mathf.Abs(
                currentJointChanges[joint]) / max_change;

            controlledJointCount++;
        }

        float normalizer = Mathf.Max(controlledJointCount, 1);

        float smoothnessPenalty =
            -smoothnessWeight * smoothnessSum / normalizer;

        float movementPenalty =
            -movementWeight * movementSum / normalizer;

        // Measure the distance between each hand and the cube.
        float rightDistance = Vector3.Distance(
            effector_1.transform.position,
            target.transform.position);

        float leftDistance = Vector3.Distance(
            effector_2.transform.position,
            target.transform.position);

        float successReward = 0f;

        // RIGHT HAND: lock on success or prolonged lack of progress.
        if (!rightHandLocked)
        {
            if (rightDistance <= reachDistance)
            {
                rightHandLocked = true;
                successReward += reachBonus;
            }
            else
            {
                if (rightDistance < bestRightDistance - noProgressEpsilon)
                {
                    bestRightDistance = rightDistance;
                    rightNoProgressSteps = 0;
                }
                else
                {
                    rightNoProgressSteps++;
                }

                if (rightNoProgressSteps >= noProgressStepsToLock)
                {
                    rightHandLocked = true;
                }
            }
        }

        // LEFT HAND: lock on success or prolonged lack of progress.
        if (!leftHandLocked)
        {
            if (leftDistance <= reachDistance)
            {
                leftHandLocked = true;
                successReward += reachBonus;
            }
            else
            {
                if (leftDistance < bestLeftDistance - noProgressEpsilon)
                {
                    bestLeftDistance = leftDistance;
                    leftNoProgressSteps = 0;
                }
                else
                {
                    leftNoProgressSteps++;
                }

                if (leftNoProgressSteps >= noProgressStepsToLock)
                {
                    leftHandLocked = true;
                }
            }
        }

        // Reward progress toward the cube.
        float progressReward = progressWeight *
            ((previousRightDistance - rightDistance) +
             (previousLeftDistance - leftDistance));

        // Encourage both hands to remain close to the cube.
        float averageDistance = (rightDistance + leftDistance) / 2f;

        float proximityPenalty =
            -distancePenaltyWeight * averageDistance;

        // Combine all reward components.
        float totalReward =
            progressReward +
            proximityPenalty +
            successReward +
            smoothnessPenalty +
            movementPenalty;

        AddReward(totalReward);

        // Update distances for the next decision.
        previousRightDistance = rightDistance;
        previousLeftDistance = leftDistance;
    }
}
