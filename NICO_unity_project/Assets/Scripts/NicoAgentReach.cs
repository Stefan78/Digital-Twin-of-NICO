using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// Trains the NICO right hand to reach a single physical target cube.
///
/// Right-hand joints can be configured individually in the Inspector:
/// - Fixed joints receive a position-drive target angle, but are not NN actions.
/// - Non-fixed joints are controlled by the neural network.
///
/// A fixed joint is NOT physically locked. Its ArticulationDrive attempts to
/// maintain the requested target; physics can still displace it depending on
/// stiffness, damping, force limit, and loads in the kinematic chain.
/// </summary>
public class NicoAgentReach : Agent
{
    [Serializable]
    public class JointControlSetting
    {
        [Tooltip("One of the right-hand ArticulationBody joints controlled by this Agent.")]
        public ArticulationBody joint;

        [Tooltip("Enable this to remove the joint from the NN action space and hold its drive target at Fixed Target Angle.")]
        public bool isFixed;

        [Tooltip("Target angle in degrees for revolute joints (or meters for prismatic joints). The ArticulationDrive attempts to maintain this value; it is not a hard physical lock.")]
        public float fixedTargetAngle;
    }

    // ---------------------------------------------------------------------
    // RIGHT-HAND JOINT CONFIGURATION
    // ---------------------------------------------------------------------

    [Header("Right-Hand Joint Configuration")]
    [Tooltip("When enabled, right-hand joints are automatically detected by name (starts with 'r_') and must not have a tag containing 'continuation'. When disabled, assign the right-hand joints manually below.")]
    [SerializeField] private bool autoDetectRightHandJoints = true;

    [Tooltip("Right-hand actuated joints. This list order defines the order of NN actions after fixed joints are removed. If auto-detection is enabled, this list is rebuilt at initialization.")]
    [SerializeField] private List<ArticulationBody> rightHandJoints = new List<ArticulationBody>();

    [Header("Per-Joint Fixed / NN Configuration")]
    [Tooltip("Add one entry for each joint you want to configure. Enable Is Fixed for any number of joints and enter its target angle. Joints not marked fixed remain NN-controlled. A configured joint must also be present in Right Hand Joints.")]
    [SerializeField] private List<JointControlSetting> jointControlSettings = new List<JointControlSetting>();

    // ---------------------------------------------------------------------
    // SINGLE CUBE / DETERMINISTIC CURRICULUM
    // ---------------------------------------------------------------------

    [Header("Single Target Cube")]
    [Tooltip("The one physical cube. Its position changes at the beginning of each episode.")]
    public GameObject target;

    [Tooltip("Right-hand end effector. Its distance to the target defines success.")]
    public GameObject rightHandEffector;

    [Header("Deterministic Target Positions")]
    [Tooltip("Total number of target positions. Position 1 is the cube's initial scene position; positions 2 onward are generated deterministically.")]
    [SerializeField, Min(1)] private int totalTargetPositions = 2;

    [Tooltip("Number of generated positions available at the beginning of training.")]
    [SerializeField, Min(1)] private int initialTargetsUnlocked = 1;

    [Tooltip("Seed used to generate the ordered target positions. The same seed and settings reproduce the same positions.")]
    [SerializeField] private int positionSeed = 12345;

    [Tooltip("Offset from the cube's initial scene position to the centre of the random-position area.")]
    [SerializeField] private Vector3 randomPositionOffsetFromFirst = new Vector3(0.3f, 0f, 0f);

    [Tooltip("Half-size of the random-position box on each axis, in world units.")]
    [SerializeField] private Vector3 randomPositionHalfExtents = new Vector3(0.1f, 0.1f, 0.1f);

    [Tooltip("Minimum distance between generated positions. If no candidate satisfies it after 100 attempts, the last candidate is used.")]
    [SerializeField, Min(0f)] private float minimumPositionSeparation = 0.05f;

    [Tooltip("Separate seed for episode target selection. Reproduces the same target-selection sequence when the episode sequence and curriculum unlocks are the same.")]
    [SerializeField] private int episodeSelectionSeed = 67890;

    private readonly List<Vector3> generatedTargetPositions = new List<Vector3>();
    private Vector3 firstTargetPosition;
    private int unlockedTargetCount;
    private DeterministicRandom episodeSelectionRandom;
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();

    // ---------------------------------------------------------------------
    // CURRICULUM / EPISODE SETTINGS
    // ---------------------------------------------------------------------

    [Header("Curriculum")]
    [SerializeField, Min(1)] private int successWindowSize = 50;
    [SerializeField, Range(0f, 1f)] private float successRateToUnlockNextTarget = 0.8f;

    [Header("Episode Length")]
    [Tooltip("Maximum number of Agent decisions in one episode. A timeout counts as a failure.")]
    [SerializeField, Min(1)] private int maxEpisodeSteps = 1000;
    private int episodeSteps;

    // ---------------------------------------------------------------------
    // REWARD SETTINGS
    // ---------------------------------------------------------------------

    [Header("Reach Reward Settings")]
    [SerializeField, Min(0f)] private float reachDistance = 0.05f;
    [SerializeField] private float progressWeight = 3f;
    [SerializeField] private float distancePenaltyWeight = 0.02f;
    [SerializeField] private float reachBonus = 1f;
    [SerializeField] private float smoothnessWeight = 0.01f;
    [SerializeField] private float movementWeight = 0.005f;

    // ---------------------------------------------------------------------
    // ACTION SETTINGS
    // ---------------------------------------------------------------------

    [Header("Action Settings")]
    [Tooltip("Maximum target increment per decision after action smoothing, in ArticulationDrive target units (degrees for revolute joints).")]
    [SerializeField, Min(0f)] private float maxAccumulatedChange = 2f;

    [Tooltip("How quickly the target increment changes in response to the NN action.")]
    [SerializeField, Min(0f)] private float actionChangeMagnitude = 0.3f;

    // ---------------------------------------------------------------------
    // STALL DETECTION
    // ---------------------------------------------------------------------

    [Header("Stall Detection")]
    [SerializeField, Min(1)] private int noProgressStepsToEndEpisode = 100;
    [SerializeField, Min(0f)] private float noProgressEpsilon = 0.001f;

    // ---------------------------------------------------------------------
    // RUNTIME DATA
    // ---------------------------------------------------------------------

    private readonly List<ArticulationBody> nnControlledJoints = new List<ArticulationBody>();
    private readonly Dictionary<ArticulationBody, float> defaultJointTargets =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, float> fixedJointTargets =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, float> currentJointChanges =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, ArticulationBody[]> continuationChains =
        new Dictionary<ArticulationBody, ArticulationBody[]>();

    private float previousDistance;
    private float bestDistance;
    private int noProgressSteps;
    private bool actionSizeWarningIssued;

    /// <summary>
    /// Small deterministic pseudo-random generator. Separate instances are used
    /// for position generation and episode target selection.
    /// </summary>
    private sealed class DeterministicRandom
    {
        private uint state;

        public DeterministicRandom(int seed)
        {
            state = unchecked((uint)seed);
            if (state == 0u)
                state = 0x6D2B79F5u;
        }

        private uint NextUInt()
        {
            uint value = state;
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            state = value;
            return value;
        }

        public float Range(float minInclusive, float maxExclusive)
        {
            float unit = (NextUInt() >> 8) * (1f / 16777216f);
            return minInclusive + (maxExclusive - minInclusive) * unit;
        }

        public int Range(int minInclusive, int maxExclusive)
        {
            if (maxExclusive <= minInclusive)
                return minInclusive;

            uint range = (uint)(maxExclusive - minInclusive);
            return minInclusive + (int)(NextUInt() % range);
        }
    }

    // ---------------------------------------------------------------------
    // INITIALIZATION
    // ---------------------------------------------------------------------

    public override void Initialize()
    {
        base.Initialize();

        if (GetComponent<ArticulationBody>() == null)
        {
            Debug.LogError("NicoAgentReach requires an ArticulationBody on the same GameObject as the Agent.");
            enabled = false;
            return;
        }

        if (target == null)
        {
            Debug.LogError("NicoAgentReach: assign the single target cube in the Target field.");
            enabled = false;
            return;
        }

        if (rightHandEffector == null)
        {
            Debug.LogError("NicoAgentReach: assign the right-hand end effector.");
            enabled = false;
            return;
        }

        PrepareRightHandJointList();
        if (rightHandJoints.Count == 0)
        {
            Debug.LogError("NicoAgentReach: no right-hand joints were found or assigned. Check the joint list or the r_ naming convention.");
            enabled = false;
            return;
        }

        CacheContinuationChains();
        PrepareFixedAndNNJointLists();
        if (!enabled)
            return;

        if (nnControlledJoints.Count == 0)
        {
            Debug.LogError("NicoAgentReach: all right-hand joints are marked fixed. At least one joint must remain NN-controlled.");
            enabled = false;
            return;
        }

        firstTargetPosition = target.transform.position;
        GenerateTargetPositionsDeterministically();
        unlockedTargetCount = Mathf.Clamp(initialTargetsUnlocked, 1, generatedTargetPositions.Count);
        episodeSelectionRandom = new DeterministicRandom(episodeSelectionSeed);

        foreach (ArticulationBody joint in nnControlledJoints)
        {
            defaultJointTargets[joint] = joint.xDrive.target;
            currentJointChanges[joint] = 0f;
        }

        LogJointConfiguration();
    }

    private void PrepareRightHandJointList()
    {
        if (autoDetectRightHandJoints)
        {
            rightHandJoints.Clear();
            ArticulationBody[] allBodies = GetComponentsInChildren<ArticulationBody>(true);

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
        else
        {
            rightHandJoints = rightHandJoints
                .Where(joint => joint != null)
                .Distinct()
                .ToList();
        }
    }

    private void CacheContinuationChains()
    {
        continuationChains.Clear();
        HashSet<ArticulationBody> primaryJoints = new HashSet<ArticulationBody>(rightHandJoints);

        foreach (ArticulationBody body in GetComponentsInChildren<ArticulationBody>(true))
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

    private void PrepareFixedAndNNJointLists()
    {
        fixedJointTargets.Clear();
        nnControlledJoints.Clear();

        HashSet<ArticulationBody> rightHandSet = new HashSet<ArticulationBody>(rightHandJoints);

        foreach (JointControlSetting setting in jointControlSettings)
        {
            if (setting == null || !setting.isFixed)
                continue;

            if (setting.joint == null)
            {
                Debug.LogError("NicoAgentReach: a joint-control entry is marked Is Fixed but has no ArticulationBody assigned.");
                enabled = false;
                return;
            }

            if (!rightHandSet.Contains(setting.joint))
            {
                Debug.LogError(
                    $"NicoAgentReach: fixed joint '{setting.joint.name}' is not in the Right Hand Joints list. " +
                    "Add it to that list or check auto-detection (name must start with 'r_' and its tag must not contain 'continuation').");
                enabled = false;
                return;
            }

            if (fixedJointTargets.ContainsKey(setting.joint))
            {
                Debug.LogWarning($"NicoAgentReach: duplicate fixed-joint setting for '{setting.joint.name}'. The first value is kept.");
                continue;
            }

            ArticulationDrive drive = setting.joint.xDrive;
            float clampedTarget = Mathf.Clamp(setting.fixedTargetAngle, drive.lowerLimit, drive.upperLimit);

            if (!Mathf.Approximately(clampedTarget, setting.fixedTargetAngle))
            {
                Debug.LogWarning(
                    $"NicoAgentReach: fixed target for '{setting.joint.name}' was clamped from " +
                    $"{setting.fixedTargetAngle} to {clampedTarget} because of its drive limits.");
            }

            fixedJointTargets.Add(setting.joint, clampedTarget);
        }

        foreach (ArticulationBody joint in rightHandJoints)
        {
            if (joint != null && !fixedJointTargets.ContainsKey(joint))
                nnControlledJoints.Add(joint);
        }
    }

    private void LogJointConfiguration()
    {
        Debug.Log("[NicoAgentReach] Right-hand joint configuration:");

        foreach (ArticulationBody joint in rightHandJoints)
        {
            if (fixedJointTargets.TryGetValue(joint, out float fixedTarget))
                Debug.Log($"[NicoAgentReach] {joint.name}: FIXED drive target = {fixedTarget:F2}");
            else
            {
                int actionIndex = nnControlledJoints.IndexOf(joint);
                Debug.Log($"[NicoAgentReach] {joint.name}: NN continuous action = {actionIndex}");
            }
        }

        Debug.Log(
            $"[NicoAgentReach] Right-hand joints: {rightHandJoints.Count}; " +
            $"fixed: {fixedJointTargets.Count}; NN-controlled: {nnControlledJoints.Count}. " +
            $"Expected Behavior Parameters: {rightHandJoints.Count + 3} vector observations, " +
            $"{nnControlledJoints.Count} continuous actions.");
    }

    // ---------------------------------------------------------------------
    // DETERMINISTIC TARGET POSITION GENERATION
    // ---------------------------------------------------------------------

    /// <summary>
    /// Position 1 is the cube's initial scene position. Positions 2 onward are
    /// sampled deterministically from positionSeed.
    /// </summary>
    private void GenerateTargetPositionsDeterministically()
    {
        generatedTargetPositions.Clear();
        generatedTargetPositions.Add(firstTargetPosition);

        int count = Mathf.Max(1, totalTargetPositions);
        DeterministicRandom positionRandom = new DeterministicRandom(positionSeed);

        Vector3 centre = firstTargetPosition + randomPositionOffsetFromFirst;
        Vector3 extents = new Vector3(
            Mathf.Abs(randomPositionHalfExtents.x),
            Mathf.Abs(randomPositionHalfExtents.y),
            Mathf.Abs(randomPositionHalfExtents.z));

        for (int i = 1; i < count; i++)
        {
            Vector3 candidate = centre;
            bool sufficientlySeparated = false;

            for (int attempt = 0; attempt < 100; attempt++)
            {
                candidate = new Vector3(
                    positionRandom.Range(centre.x - extents.x, centre.x + extents.x),
                    positionRandom.Range(centre.y - extents.y, centre.y + extents.y),
                    positionRandom.Range(centre.z - extents.z, centre.z + extents.z));

                sufficientlySeparated = generatedTargetPositions.All(
                    existing => Vector3.Distance(existing, candidate) >= minimumPositionSeparation);

                if (sufficientlySeparated)
                    break;
            }

            if (!sufficientlySeparated)
            {
                Debug.LogWarning(
                    "NicoAgentReach: could not generate a position satisfying Minimum Position Separation " +
                    "after 100 attempts. The last candidate will be used. Increase the random-position area " +
                    "or reduce Minimum Position Separation.");
            }

            generatedTargetPositions.Add(candidate);
        }

        for (int i = 0; i < generatedTargetPositions.Count; i++)
        {
            Debug.Log($"[Reach Curriculum] Target position {i + 1}/{generatedTargetPositions.Count}: {generatedTargetPositions[i]}");
        }
    }

    // ---------------------------------------------------------------------
    // JOINT DRIVE / RESET HELPERS
    // ---------------------------------------------------------------------

    private static void SetDriveTarget(ArticulationBody joint, float targetValue)
    {
        if (joint == null)
            return;

        ArticulationDrive drive = joint.xDrive;
        drive.target = Mathf.Clamp(targetValue, drive.lowerLimit, drive.upperLimit);
        joint.xDrive = drive;
    }

    /// <summary>
    /// Applies the drive target to a primary joint and any tagged continuation
    /// joints below it. Continuation joints are not separate NN actions.
    /// </summary>
    private void SetJointTarget(ArticulationBody joint, float targetValue)
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

    /// <summary>
    /// Resets a joint to a known pose at the start of an episode. This is called
    /// only at episode boundaries; fixed joints are not teleported or overwritten
    /// every simulation step. During the episode their drives respond to physics.
    /// </summary>
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

    private static void ResetOneJointState(ArticulationBody joint, float targetValue)
    {
        if (joint == null)
            return;

        SetDriveTarget(joint, targetValue);

        if (joint.dofCount > 0)
        {
            // ArticulationDrive.target uses degrees for revolute joints, while
            // jointPosition uses radians. Prismatic coordinates use meters in both.
            float reducedCoordinate = joint.jointType == ArticulationJointType.PrismaticJoint
                ? targetValue
                : targetValue * Mathf.Deg2Rad;

            joint.jointPosition = new ArticulationReducedSpace(reducedCoordinate);
        }

        joint.jointVelocity = new ArticulationReducedSpace(0f);
        joint.linearVelocity = Vector3.zero;
        joint.angularVelocity = Vector3.zero;
    }

    /// <summary>
    /// Returns the current physical joint coordinate in the same units as xDrive
    /// (degrees for revolute joints, meters for prismatic joints).
    /// </summary>
    private static float GetCurrentJointCoordinate(ArticulationBody joint)
    {
        if (joint == null || joint.dofCount == 0)
            return 0f;

        float coordinate = joint.jointPosition[0];
        if (joint.jointType == ArticulationJointType.RevoluteJoint)
            coordinate *= Mathf.Rad2Deg;

        return coordinate;
    }

    private static float NormalizeJointCoordinate(ArticulationBody joint)
    {
        ArticulationDrive drive = joint.xDrive;
        float range = drive.upperLimit - drive.lowerLimit;

        if (Mathf.Abs(range) < 0.0001f)
            return 0f;

        float coordinate = GetCurrentJointCoordinate(joint);
        return Mathf.Clamp01((coordinate - drive.lowerLimit) / range);
    }

    // ---------------------------------------------------------------------
    // EPISODES / OBSERVATIONS / ACTIONS
    // ---------------------------------------------------------------------

    public override void OnEpisodeBegin()
    {
        episodeSteps = 0;
        noProgressSteps = 0;
        actionSizeWarningIssued = false;

        // Choose one currently unlocked position using the independent seeded stream.
        int selectedIndex = episodeSelectionRandom.Range(0, unlockedTargetCount);
        target.transform.position = generatedTargetPositions[selectedIndex];

        // Reset configured fixed joints to their requested starting pose.
        // After this reset, their ArticulationDrives (not the NN) maintain the targets.
        foreach (KeyValuePair<ArticulationBody, float> fixedJoint in fixedJointTargets)
            ResetJointState(fixedJoint.Key, fixedJoint.Value);

        // Reset NN-controlled joints to the scene's initial drive targets.
        foreach (ArticulationBody joint in nnControlledJoints)
        {
            ResetJointState(joint, defaultJointTargets[joint]);
            currentJointChanges[joint] = 0f;
        }

        previousDistance = Vector3.Distance(
            rightHandEffector.transform.position, target.transform.position);
        bestDistance = previousDistance;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe the CURRENT PHYSICAL position of every configured right-hand joint,
        // including fixed joints. This lets the policy observe physics-induced deviations.
        foreach (ArticulationBody joint in rightHandJoints)
            sensor.AddObservation(NormalizeJointCoordinate(joint));

        // Relative target position from the right-hand end effector (3 values).
        sensor.AddObservation(target.transform.position - rightHandEffector.transform.position);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        episodeSteps++;

        int availableActions = actions.ContinuousActions.Length;
        if (availableActions != nnControlledJoints.Count && !actionSizeWarningIssued)
        {
            Debug.LogError(
                $"NicoAgentReach: Behavior Parameters provides {availableActions} continuous actions, " +
                $"but {nnControlledJoints.Count} joints are marked for NN control. Set Continuous Actions " +
                $"to {nnControlledJoints.Count}.");
            actionSizeWarningIssued = true;
        }

        float smoothnessSum = 0f;
        float movementSum = 0f;
        int controlledJointCount = 0;
        int actionCount = Mathf.Min(availableActions, nnControlledJoints.Count);

        // Only non-fixed joints appear in nnControlledJoints and receive NN actions.
        for (int i = 0; i < actionCount; i++)
        {
            ArticulationBody joint = nnControlledJoints[i];
            float action = Mathf.Clamp(actions.ContinuousActions[i], -1f, 1f);

            // Preserve the incremental controller used by the original agent:
            // the NN action changes a bounded target-step command, and that
            // command is added to the CURRENT drive target each decision.
            // This lets the joint travel through its full drive range. It is
            // deliberately not restricted to a tiny offset around its
            // episode-start target.
            float previousChange = currentJointChanges[joint];
            currentJointChanges[joint] = Mathf.Clamp(
                previousChange + action * actionChangeMagnitude,
                -maxAccumulatedChange,
                maxAccumulatedChange);

            float newTarget = Mathf.Clamp(
                joint.xDrive.target + currentJointChanges[joint],
                joint.xDrive.lowerLimit,
                joint.xDrive.upperLimit);

            SetJointTarget(joint, newTarget);

            float normalizer = Mathf.Max(maxAccumulatedChange, 0.0001f);
            smoothnessSum += Mathf.Abs(currentJointChanges[joint] - previousChange) / normalizer;
            movementSum += Mathf.Abs(currentJointChanges[joint]) / normalizer;
            controlledJointCount++;
        }

        float jointNormalizer = Mathf.Max(controlledJointCount, 1);
        float smoothnessPenalty = -smoothnessWeight * smoothnessSum / jointNormalizer;
        float movementPenalty = -movementWeight * movementSum / jointNormalizer;

        float distance = Vector3.Distance(
            rightHandEffector.transform.position, target.transform.position);

        float progressReward = progressWeight * (previousDistance - distance);
        float proximityPenalty = -distancePenaltyWeight * distance;
        float successReward = distance <= reachDistance ? reachBonus : 0f;

        AddReward(progressReward + proximityPenalty + successReward + smoothnessPenalty + movementPenalty);
        previousDistance = distance;

        if (distance < bestDistance - noProgressEpsilon)
        {
            bestDistance = distance;
            noProgressSteps = 0;
        }
        else
        {
            noProgressSteps++;
        }

        if (distance <= reachDistance)
        {
            FinishEpisode(true);
            return;
        }

        if (noProgressSteps >= noProgressStepsToEndEpisode)
        {
            FinishEpisode(false);
            return;
        }

        if (episodeSteps >= maxEpisodeSteps)
        {
            FinishEpisode(false);
            return;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Neutral actions for Heuristic/manual mode. Training uses the NN policy.
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        for (int i = 0; i < continuousActions.Length; i++)
            continuousActions[i] = 0f;
    }

    // ---------------------------------------------------------------------
    // CURRICULUM PROGRESSION
    // ---------------------------------------------------------------------

    private void FinishEpisode(bool success)
    {
        RecordEpisodeResult(success);
        EndEpisode();
    }

    private void RecordEpisodeResult(bool success)
    {
        int windowSize = Mathf.Max(1, successWindowSize);

        recentEpisodeResults.Enqueue(success);
        while (recentEpisodeResults.Count > windowSize)
            recentEpisodeResults.Dequeue();

        if (recentEpisodeResults.Count < windowSize || unlockedTargetCount >= generatedTargetPositions.Count)
            return;

        int successes = recentEpisodeResults.Count(result => result);
        float successRate = successes / (float)recentEpisodeResults.Count;

        if (successRate >= successRateToUnlockNextTarget)
        {
            int oldTargetCount = unlockedTargetCount;
            unlockedTargetCount = Mathf.Min(unlockedTargetCount + 1, generatedTargetPositions.Count);
            recentEpisodeResults.Clear();

            Debug.Log(
                $"[Reach Curriculum] Success rate {successRate:P0} over the last {windowSize} episodes. " +
                $"Unlocked {unlockedTargetCount}/{generatedTargetPositions.Count} target positions (was {oldTargetCount}).");
        }
    }
}
