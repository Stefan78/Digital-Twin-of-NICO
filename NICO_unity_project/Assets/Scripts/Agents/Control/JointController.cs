using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Discovers and configures the NICO right-hand articulation joints, manages
/// fixed/NN-controlled joint partitions, and applies/reset articulation targets.
/// This is an ordinary C# helper class, not a MonoBehaviour.
/// </summary>
public sealed class JointController
{
    private readonly List<ArticulationBody> rightHandJoints = new List<ArticulationBody>();
    private readonly List<ArticulationBody> nnControlledJoints = new List<ArticulationBody>();

    private readonly Dictionary<ArticulationBody, float> defaultJointTargets =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, float> fixedJointTargets =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, float> currentJointChanges =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, ArticulationBody[]> continuationChains =
        new Dictionary<ArticulationBody, ArticulationBody[]>();

    public List<ArticulationBody> RightHandJoints => rightHandJoints;
    public List<ArticulationBody> NNControlledJoints => nnControlledJoints;
    public Dictionary<ArticulationBody, float> CurrentJointChanges => currentJointChanges;
    public int FixedJointCount => fixedJointTargets.Count;

    public bool Initialize(
        Transform agentRoot,
        bool autoDetectRightHandJoints,
        List<ArticulationBody> inspectorJointList,
        List<NicoAgentReach2.JointControlSetting> jointControlSettings,
        out string errorMessage)
    {
        errorMessage = null;

        PrepareRightHandJointList(agentRoot, autoDetectRightHandJoints, inspectorJointList);
        CacheContinuationChains(agentRoot);

        if (!PrepareFixedAndNNJointLists(jointControlSettings, out errorMessage))
            return false;

        if (rightHandJoints.Count == 0)
            return true; // The Agent provides the user-facing no-joints error.

        if (nnControlledJoints.Count == 0)
            return true; // The Agent provides the user-facing all-fixed error.

        foreach (ArticulationBody joint in nnControlledJoints)
        {
            defaultJointTargets[joint] = joint.xDrive.target;
            currentJointChanges[joint] = 0f;
        }

        return true;
    }

    private void PrepareRightHandJointList(
        Transform agentRoot,
        bool autoDetect,
        List<ArticulationBody> inspectorJointList)
    {
        rightHandJoints.Clear();

        if (autoDetect)
        {
            ArticulationBody[] allBodies = agentRoot.GetComponentsInChildren<ArticulationBody>(true);
            foreach (ArticulationBody body in allBodies)
            {
                if (body == null)
                    continue;

                string jointName = body.name.ToLowerInvariant();
                string jointTag = body.tag.ToLowerInvariant();
                if (jointName.StartsWith("r_") && !jointTag.Contains("continuation"))
                    rightHandJoints.Add(body);
            }
        }
        else if (inspectorJointList != null)
        {
            rightHandJoints.AddRange(
                inspectorJointList.Where(joint => joint != null).Distinct());
        }
    }

    private void CacheContinuationChains(Transform agentRoot)
    {
        continuationChains.Clear();
        HashSet<ArticulationBody> primaryJoints = new HashSet<ArticulationBody>(rightHandJoints);

        foreach (ArticulationBody body in agentRoot.GetComponentsInChildren<ArticulationBody>(true))
        {
            if (body == null || !body.tag.ToLowerInvariant().Contains("base"))
                continue;

            ArticulationBody[] descendants = body.GetComponentsInChildren<ArticulationBody>(true)
                .Where(child => child != body &&
                                child.tag.ToLowerInvariant().Contains("continuation") &&
                                !primaryJoints.Contains(child))
                .ToArray();

            if (descendants.Length > 0)
                continuationChains[body] = descendants;
        }
    }

    private bool PrepareFixedAndNNJointLists(
        List<NicoAgentReach2.JointControlSetting> jointControlSettings,
        out string errorMessage)
    {
        errorMessage = null;
        fixedJointTargets.Clear();
        nnControlledJoints.Clear();

        HashSet<ArticulationBody> rightHandSet = new HashSet<ArticulationBody>(rightHandJoints);

        if (jointControlSettings != null)
        {
            foreach (NicoAgentReach2.JointControlSetting setting in jointControlSettings)
            {
                if (setting == null || !setting.isFixed)
                    continue;

                if (setting.joint == null)
                {
                    errorMessage = "NicoAgentReach: a joint-control entry is marked Is Fixed but has no ArticulationBody assigned.";
                    return false;
                }

                if (!rightHandSet.Contains(setting.joint))
                {
                    errorMessage =
                        $"NicoAgentReach: fixed joint '{setting.joint.name}' is not in the Right Hand Joints list. " +
                        "Add it to that list or check auto-detection (name must start with 'r_' and its tag must not contain 'continuation').";
                    return false;
                }

                if (fixedJointTargets.ContainsKey(setting.joint))
                {
                    Debug.LogWarning(
                        $"NicoAgentReach: duplicate fixed-joint setting for '{setting.joint.name}'. The first value is kept.");
                    continue;
                }

                ArticulationDrive drive = setting.joint.xDrive;
                float clampedTarget = Mathf.Clamp(
                    setting.fixedTargetAngle,
                    drive.lowerLimit,
                    drive.upperLimit);

                if (!Mathf.Approximately(clampedTarget, setting.fixedTargetAngle))
                {
                    Debug.LogWarning(
                        $"NicoAgentReach: fixed target for '{setting.joint.name}' was clamped from " +
                        $"{setting.fixedTargetAngle} to {clampedTarget} because of its drive limits.");
                }

                fixedJointTargets.Add(setting.joint, clampedTarget);
            }
        }

        foreach (ArticulationBody joint in rightHandJoints)
        {
            if (joint != null && !fixedJointTargets.ContainsKey(joint))
                nnControlledJoints.Add(joint);
        }

        return true;
    }

    public bool TryGetFixedTarget(ArticulationBody joint, out float target)
    {
        return fixedJointTargets.TryGetValue(joint, out target);
    }

    public void ResetEpisodeJoints()
    {
        // Fixed joints are reset first, followed by NN-controlled joints, as before.
        foreach (KeyValuePair<ArticulationBody, float> fixedJoint in fixedJointTargets)
            ResetJointState(fixedJoint.Key, fixedJoint.Value);

        foreach (ArticulationBody joint in nnControlledJoints)
        {
            ResetJointState(joint, defaultJointTargets[joint]);
            currentJointChanges[joint] = 0f;
        }
    }

    public void SetJointTarget(ArticulationBody joint, float targetValue)
    {
        SetDriveTarget(joint, targetValue);

        if (continuationChains.TryGetValue(joint, out ArticulationBody[] chain))
        {
            foreach (ArticulationBody continuation in chain)
            {
                float ratio = continuation.tag.ToLowerInvariant().Contains("thumb") ? 0.25f : 1f;
                SetDriveTarget(continuation, targetValue * ratio);
            }
        }
    }

    private void ResetJointState(ArticulationBody joint, float targetValue)
    {
        if (joint == null)
            return;

        ResetOneJointState(joint, targetValue);

        if (continuationChains.TryGetValue(joint, out ArticulationBody[] chain))
        {
            foreach (ArticulationBody continuation in chain)
            {
                float ratio = continuation.tag.ToLowerInvariant().Contains("thumb") ? 0.25f : 1f;
                ResetOneJointState(continuation, targetValue * ratio);
            }
        }
    }

    private static void SetDriveTarget(ArticulationBody joint, float targetValue)
    {
        if (joint == null)
            return;

        ArticulationDrive drive = joint.xDrive;
        drive.target = Mathf.Clamp(targetValue, drive.lowerLimit, drive.upperLimit);
        joint.xDrive = drive;
    }

    private static void ResetOneJointState(ArticulationBody joint, float targetValue)
    {
        if (joint == null)
            return;

        SetDriveTarget(joint, targetValue);

        if (joint.dofCount > 0)
        {
            // ArticulationDrive.target uses degrees for revolute joints,
            // while ArticulationBody.jointPosition uses radians.
            float reducedCoordinate = joint.jointType == ArticulationJointType.PrismaticJoint
                ? targetValue
                : targetValue * Mathf.Deg2Rad;

            joint.jointPosition = new ArticulationReducedSpace(reducedCoordinate);
        }

        joint.jointVelocity = new ArticulationReducedSpace(0f);
        joint.linearVelocity = Vector3.zero;
        joint.angularVelocity = Vector3.zero;
    }

    public float NormalizeJointCoordinate(ArticulationBody joint)
    {
        ArticulationDrive drive = joint.xDrive;
        float range = drive.upperLimit - drive.lowerLimit;
        if (Mathf.Abs(range) < 0.0001f)
            return 0f;

        float coordinate = GetCurrentJointCoordinate(joint);
        return Mathf.Clamp01((coordinate - drive.lowerLimit) / range);
    }

    private static float GetCurrentJointCoordinate(ArticulationBody joint)
    {
        if (joint == null || joint.dofCount == 0)
            return 0f;

        float coordinate = joint.jointPosition[0];
        if (joint.jointType == ArticulationJointType.RevoluteJoint)
            coordinate *= Mathf.Rad2Deg;

        return coordinate;
    }
}
