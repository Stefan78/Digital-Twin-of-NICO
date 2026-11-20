using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// Trains the NICO right hand to reach one physical target cube.
///
/// The target cube is sampled uniformly inside a BoxCollider assigned as
/// Target Spawn Volume. The spawn-volume collider may be disabled; its
/// center and size are still used as geometric data.
///
/// Curriculum stages are centered, progressively larger sub-regions of the
/// full spawn volume. Each stage is unlocked after the configured recent
/// success-rate threshold is reached. After first reaching the target, the NN
/// can continue controlling the hand for a configurable post-reach interval.
///
/// Right-hand joints can be configured individually in the Inspector:
/// - Fixed joints receive a drive target but are not NN actions.
/// - Remaining joints are controlled by the neural network.
/// - All joint positions and all joint accumulated-change values are observed,
///   so the observation count stays constant when the fixed-joint selection changes.
/// - The policy also observes whether it is in the post-reach holding phase.
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

        [Tooltip("Target angle in degrees for revolute joints (or meters for prismatic joints). The drive attempts to maintain this value; it is not a hard physical lock.")]
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
    // TARGET AND SPAWN VOLUME
    // ---------------------------------------------------------------------

    [Header("Target")]
    [Tooltip("The one physical target cube. Its world position is changed at the beginning of each episode.")]
    [SerializeField] private GameObject target;

    [Tooltip("Right-hand end effector. Its distance to the target cube defines success.")]
    [SerializeField] private GameObject rightHandEffector;

    [Header("Target Spawn Volume")]
    [Tooltip("Assign the visible cube that defines the valid target workspace. It must have a BoxCollider on the same GameObject. The BoxCollider can be disabled to prevent physical interactions; this script still reads its center and size.")]
    [SerializeField] private GameObject targetSpawnVolume;

    [Tooltip("Extra clearance between the target's collider and each spawn-volume boundary, in TargetSpawnVolume local units. Set to 0 for no extra margin.")]
    [SerializeField, Min(0f)] private float spawnBoundaryMarginLocal = 0.005f;

    [Tooltip("Seed for the deterministic sequence of target positions. Same seed, spawn-volume geometry, and episode order reproduce the same sequence.")]
    [SerializeField] private int spawnPositionSeed = 12345;

    private BoxCollider targetSpawnVolumeCollider;
    private DeterministicRandom spawnPositionRandom;
    private Vector3 validTargetPivotMinLocal;
    private Vector3 validTargetPivotMaxLocal;
    private Vector3 validTargetPivotCenterLocal;
    private Vector3 validTargetPivotHalfRangeLocal;
    private bool spawnVolumeGeometryReady;

    // ---------------------------------------------------------------------
    // AREA-BASED CURRICULUM
    // ---------------------------------------------------------------------

    [Header("Area-Based Curriculum")]
    [Tooltip("Each entry defines the fraction of the FULL valid target-center range available at that curriculum stage, independently for X, Y and Z. Stages should grow or stay equal in every axis. Defaults: central 25%, 50%, 75%, then the full volume.")]
    [SerializeField]
    private List<Vector3> curriculumRegionFractions = new List<Vector3>
    {
        new Vector3(0.25f, 0.25f, 0.25f),
        new Vector3(0.50f, 0.50f, 0.50f),
        new Vector3(0.75f, 0.75f, 0.75f),
        new Vector3(1.00f, 1.00f, 1.00f)
    };

    [Tooltip("Number of curriculum stages available at the beginning. Usually 1 for training; set to the total number of stages to sample the full volume immediately.")]
    [SerializeField, Min(1)] private int initialCurriculumStagesUnlocked = 1;

    [Tooltip("When enabled, the success-rate window unlocks progressively larger regions. Disable for inference/evaluation to keep the selected region fixed.")]
    [SerializeField] private bool enableCurriculumProgression = true;

    [Tooltip("Number of completed episodes used to evaluate whether the next, larger region should be unlocked.")]
    [SerializeField, Min(1)] private int successWindowSize = 50;

    [Tooltip("Unlock the next, larger spawn region when the recent success rate reaches this value.")]
    [SerializeField, Range(0f, 1f)] private float successRateToUnlockNextRegion = 0.8f;

    private int unlockedCurriculumStageCount;
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();

    // ---------------------------------------------------------------------
    // EPISODE SETTINGS
    // ---------------------------------------------------------------------

    [Header("Episode Length")]
    [Tooltip("Maximum number of Agent decisions allowed to reach the target. After reaching, the configured Post-Reach Steps are added as an extra continuation phase.")]
    [SerializeField, Min(1)] private int maxEpisodeSteps = 1000;
    private int episodeSteps;

    // ---------------------------------------------------------------------
    // POST-REACH BEHAVIOUR
    // ---------------------------------------------------------------------

    [Header("Post-Reach Behaviour")]
    [Tooltip("Number of additional Agent decisions to continue controlling the hand after it first reaches the target. Set to 0 to end the episode immediately on reaching, as in the original behaviour.")]
    [SerializeField, Min(0)] private int postReachSteps = 100;

    [Tooltip("Distance penalty applied on every post-reach decision. This encourages the hand to remain close to the target while the NN continues controlling it. Start around 0.2 and adjust after observing training.")]
    [SerializeField, Min(0f)] private float postReachDistancePenaltyWeight = 0.2f;

    [Tooltip("Adds one observation: 0 while approaching and 1 during post-reach continuation. Recommended for new training (observation size increases by one). Disable only when testing a legacy model trained without this observation.")]
    [SerializeField] private bool includePostReachPhaseObservation = true;

    private bool hasReachedTarget;
    private int postReachStepsElapsed;

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
    [Tooltip("Maximum accumulated target-step command per decision, in ArticulationDrive target units (degrees for revolute joints). This limits the per-decision target increment, not the total joint range.")]
    [SerializeField, Min(0f)] private float maxAccumulatedChange = 2f;

    [Tooltip("How quickly the accumulated target-step command changes in response to the NN action.")]
    [SerializeField, Min(0f)] private float actionChangeMagnitude = 0.3f;

    // ---------------------------------------------------------------------
    // STALL DETECTION
    // ---------------------------------------------------------------------

    [Header("Stall Detection")]
    [SerializeField, Min(1)] private int noProgressStepsToEndEpisode = 100;
    [SerializeField, Min(0f)] private float noProgressEpsilon = 0.001f;

    // ---------------------------------------------------------------------
    // RUNTIME JOINT DATA
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
    /// Small deterministic pseudo-random generator. It avoids depending on
    /// UnityEngine.Random and makes the target-position sequence repeatable.
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
            if (maxExclusive <= minInclusive)
                return minInclusive;

            float unit = (NextUInt() >> 8) * (1f / 16777216f);
            return minInclusive + (maxExclusive - minInclusive) * unit;
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

        if (!InitializeSpawnVolume())
        {
            enabled = false;
            return;
        }

        ValidateCurriculumStages();

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

        spawnPositionRandom = new DeterministicRandom(spawnPositionSeed);
        unlockedCurriculumStageCount = Mathf.Clamp(
            initialCurriculumStagesUnlocked, 1, curriculumRegionFractions.Count);

        foreach (ArticulationBody joint in nnControlledJoints)
        {
            defaultJointTargets[joint] = joint.xDrive.target;
            currentJointChanges[joint] = 0f;
        }

        LogJointConfiguration();
        LogSpawnVolumeConfiguration();
    }

    private bool InitializeSpawnVolume()
    {
        if (targetSpawnVolume == null)
        {
            Debug.LogError("NicoAgentReach: assign the TargetSpawnVolume cube GameObject in the Inspector.");
            return false;
        }

        if (targetSpawnVolume == target)
        {
            Debug.LogError("NicoAgentReach: Target and Target Spawn Volume must be two different GameObjects.");
            return false;
        }

        if (targetSpawnVolume.transform.IsChildOf(target.transform))
        {
            Debug.LogError(
                "NicoAgentReach: TargetSpawnVolume cannot be a child of the moving Target object, " +
                "because it would move whenever the target is repositioned.");
            return false;
        }

        targetSpawnVolumeCollider = targetSpawnVolume.GetComponent<BoxCollider>();
        if (targetSpawnVolumeCollider == null)
        {
            Debug.LogError("NicoAgentReach: TargetSpawnVolume must have a BoxCollider on the same GameObject.");
            return false;
        }

        if (targetSpawnVolumeCollider.size.x <= 0f ||
            targetSpawnVolumeCollider.size.y <= 0f ||
            targetSpawnVolumeCollider.size.z <= 0f)
        {
            Debug.LogError("NicoAgentReach: TargetSpawnVolume BoxCollider must have positive size on all axes.");
            return false;
        }

        CacheTargetColliderOffsetsAndCalculateValidRange();

        if (validTargetPivotMinLocal.x > validTargetPivotMaxLocal.x ||
            validTargetPivotMinLocal.y > validTargetPivotMaxLocal.y ||
            validTargetPivotMinLocal.z > validTargetPivotMaxLocal.z)
        {
            Debug.LogError(
                "NicoAgentReach: the target cube is too large to fit inside TargetSpawnVolume " +
                "with the current Spawn Boundary Margin. Enlarge the volume or reduce the margin.");
            return false;
        }

        spawnVolumeGeometryReady = true;
        return true;
    }

    private void CacheTargetColliderOffsetsAndCalculateValidRange()
    {
        Vector3 offsetMin = Vector3.zero;
        Vector3 offsetMax = Vector3.zero;
        bool foundAnyBounds = false;

        Collider[] colliders = target.GetComponentsInChildren<Collider>(true);

        foreach (Collider targetCollider in colliders)
        {
            if (targetCollider == null)
                continue;

            BoxCollider box = targetCollider as BoxCollider;
            if (box != null)
            {
                Vector3 half = box.size * 0.5f;

                for (int corner = 0; corner < 8; corner++)
                {
                    Vector3 sign = new Vector3(
                        (corner & 1) == 0 ? -1f : 1f,
                        (corner & 2) == 0 ? -1f : 1f,
                        (corner & 4) == 0 ? -1f : 1f);

                    Vector3 pointInBoxLocal = box.center + Vector3.Scale(half, sign);
                    Vector3 pointWorld = box.transform.TransformPoint(pointInBoxLocal);
                    Vector3 offsetInSpawnLocal = targetSpawnVolume.transform.InverseTransformVector(
                        pointWorld - target.transform.position);

                    EncapsulateOffset(offsetInSpawnLocal, ref offsetMin, ref offsetMax, ref foundAnyBounds);
                }
            }
            else if (targetCollider.enabled)
            {
                // Conservative fallback for non-box collider shapes: use their world AABB corners.
                AddWorldBoundsCorners(targetCollider.bounds, ref offsetMin, ref offsetMax, ref foundAnyBounds);
            }
        }

        // If the target has no Collider components, use its Renderer bounds so the visible
        // target still fits inside the volume. This is a conservative world-AABB approximation.
        if (!foundAnyBounds)
        {
            Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
            foreach (Renderer targetRenderer in renderers)
            {
                if (targetRenderer != null && targetRenderer.enabled)
                    AddWorldBoundsCorners(targetRenderer.bounds, ref offsetMin, ref offsetMax, ref foundAnyBounds);
            }
        }

        if (!foundAnyBounds)
        {
            offsetMin = Vector3.zero;
            offsetMax = Vector3.zero;
            Debug.LogWarning(
                "NicoAgentReach: no enabled target Collider or Renderer bounds were found. " +
                "The target pivot alone will be constrained to the spawn volume.");
        }

        Vector3 volumeMin = targetSpawnVolumeCollider.center - targetSpawnVolumeCollider.size * 0.5f;
        Vector3 volumeMax = targetSpawnVolumeCollider.center + targetSpawnVolumeCollider.size * 0.5f;

        Vector3 margin = Vector3.one * spawnBoundaryMarginLocal;
        volumeMin += margin;
        volumeMax -= margin;

        // A target pivot is valid only if every target-collider corner remains inside the volume.
        validTargetPivotMinLocal = volumeMin - offsetMin;
        validTargetPivotMaxLocal = volumeMax - offsetMax;
        validTargetPivotCenterLocal = (validTargetPivotMinLocal + validTargetPivotMaxLocal) * 0.5f;
        validTargetPivotHalfRangeLocal = (validTargetPivotMaxLocal - validTargetPivotMinLocal) * 0.5f;
    }

    private void AddWorldBoundsCorners(
        Bounds worldBounds,
        ref Vector3 offsetMin,
        ref Vector3 offsetMax,
        ref bool foundAnyBounds)
    {
        Vector3 half = worldBounds.extents;

        for (int corner = 0; corner < 8; corner++)
        {
            Vector3 sign = new Vector3(
                (corner & 1) == 0 ? -1f : 1f,
                (corner & 2) == 0 ? -1f : 1f,
                (corner & 4) == 0 ? -1f : 1f);

            Vector3 pointWorld = worldBounds.center + Vector3.Scale(half, sign);
            Vector3 offsetInSpawnLocal = targetSpawnVolume.transform.InverseTransformVector(
                pointWorld - target.transform.position);

            EncapsulateOffset(offsetInSpawnLocal, ref offsetMin, ref offsetMax, ref foundAnyBounds);
        }
    }

    private static void EncapsulateOffset(
        Vector3 offset,
        ref Vector3 offsetMin,
        ref Vector3 offsetMax,
        ref bool initialized)
    {
        if (!initialized)
        {
            offsetMin = offset;
            offsetMax = offset;
            initialized = true;
            return;
        }

        offsetMin = Vector3.Min(offsetMin, offset);
        offsetMax = Vector3.Max(offsetMax, offset);
    }

    private void ValidateCurriculumStages()
    {
        if (curriculumRegionFractions == null)
            curriculumRegionFractions = new List<Vector3>();

        if (curriculumRegionFractions.Count == 0)
            curriculumRegionFractions.Add(Vector3.one);

        Vector3 previous = Vector3.zero;

        for (int i = 0; i < curriculumRegionFractions.Count; i++)
        {
            Vector3 requested = curriculumRegionFractions[i];
            Vector3 stage = new Vector3(
                Mathf.Clamp01(requested.x),
                Mathf.Clamp01(requested.y),
                Mathf.Clamp01(requested.z));

            if (i > 0)
            {
                Vector3 expanded = new Vector3(
                    Mathf.Max(stage.x, previous.x),
                    Mathf.Max(stage.y, previous.y),
                    Mathf.Max(stage.z, previous.z));

                if (expanded != stage)
                {
                    Debug.LogWarning(
                        $"NicoAgentReach: curriculum region {i + 1} was smaller than the previous stage " +
                        "on at least one axis. It has been expanded to avoid curriculum regression.");
                    stage = expanded;
                }
            }

            curriculumRegionFractions[i] = stage;
            previous = stage;
        }
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

        int expectedObservations = rightHandJoints.Count * 2 + 3 +
            (includePostReachPhaseObservation ? 1 : 0);
        Debug.Log(
            $"[NicoAgentReach] Right-hand joints: {rightHandJoints.Count}; " +
            $"fixed: {fixedJointTargets.Count}; NN-controlled: {nnControlledJoints.Count}. " +
            $"Expected Behavior Parameters: {expectedObservations} vector observations, " +
            $"{nnControlledJoints.Count} continuous actions.");
    }

    private void LogSpawnVolumeConfiguration()
    {
        Debug.Log(
            "[NicoAgentReach] TargetSpawnVolume ready. " +
            $"Valid target-pivot range in volume-local coordinates: min={validTargetPivotMinLocal}, " +
            $"max={validTargetPivotMaxLocal}. " +
            $"Spawn volume BoxCollider enabled={targetSpawnVolumeCollider.enabled} (it may remain disabled).");

        for (int i = 0; i < curriculumRegionFractions.Count; i++)
        {
            Debug.Log(
                $"[NicoAgentReach] Curriculum region {i + 1}/{curriculumRegionFractions.Count}: " +
                $"fraction={curriculumRegionFractions[i]}");
        }
    }

    // ---------------------------------------------------------------------
    // TARGET POSITION SAMPLING INSIDE THE SPAWN VOLUME
    // ---------------------------------------------------------------------

    private Vector3 GetCurrentCurriculumRegionFraction()
    {
        int stageIndex = Mathf.Clamp(
            unlockedCurriculumStageCount - 1,
            0,
            curriculumRegionFractions.Count - 1);

        return curriculumRegionFractions[stageIndex];
    }

    private Vector3 GenerateRandomTargetWorldPosition()
    {
        Vector3 fraction = GetCurrentCurriculumRegionFraction();
        Vector3 regionHalfRange = Vector3.Scale(validTargetPivotHalfRangeLocal, fraction);

        Vector3 localTargetPivot = validTargetPivotCenterLocal + new Vector3(
            spawnPositionRandom.Range(-regionHalfRange.x, regionHalfRange.x),
            spawnPositionRandom.Range(-regionHalfRange.y, regionHalfRange.y),
            spawnPositionRandom.Range(-regionHalfRange.z, regionHalfRange.z));

        return targetSpawnVolume.transform.TransformPoint(localTargetPivot);
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
    /// Resets a joint at episode boundaries. Fixed joints are then maintained
    /// by their articulation drives and are not overwritten every simulation step.
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

    /// <summary>
    /// Returns the current physical joint coordinate in xDrive units:
    /// degrees for revolute joints and meters for prismatic joints.
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
        hasReachedTarget = false;
        postReachStepsElapsed = 0;

        // Sample a new target position uniformly inside the currently unlocked
        // curriculum sub-cuboid. The one target GameObject is reused every episode.
        target.transform.position = GenerateRandomTargetWorldPosition();

        // Reset configured fixed joints to their requested starting pose.
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
        // 1. Observe the CURRENT PHYSICAL coordinate of every right-hand joint.
        // This includes fixed joints and makes physics-induced deviations observable.
        foreach (ArticulationBody joint in rightHandJoints)
            sensor.AddObservation(NormalizeJointCoordinate(joint));

        // 2. Observe accumulated action-change commands for ALL right-hand joints.
        // Fixed joints have no NN command, so TryGetValue returns zero for them.
        float changeScale = Mathf.Max(maxAccumulatedChange, 0.0001f);
        foreach (ArticulationBody joint in rightHandJoints)
        {
            currentJointChanges.TryGetValue(joint, out float accumulatedChange);
            sensor.AddObservation(Mathf.Clamp(accumulatedChange / changeScale, -1f, 1f));
        }

        // 3. Relative target position from the right-hand end effector (3 values).
        sensor.AddObservation(
            target.transform.position - rightHandEffector.transform.position);

        // 4. Optional phase indicator: 0 while approaching, 1 after first reaching.
        // This lets the policy distinguish approach behaviour from post-reach holding,
        // even if the hand temporarily drifts outside the reach radius. Disable only
        // when the assigned legacy model expects the previous observation size.
        if (includePostReachPhaseObservation)
            sensor.AddObservation(hasReachedTarget ? 1f : 0f);
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

        // The NN continues controlling every non-fixed joint both before and
        // after reaching the target. No joints are frozen in the post-reach phase.
        for (int i = 0; i < actionCount; i++)
        {
            ArticulationBody joint = nnControlledJoints[i];
            float action = Mathf.Clamp(actions.ContinuousActions[i], -1f, 1f);

            // Incremental controller: NN output changes a bounded target-step command.
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

        float progressReward = 0f;
        float proximityPenalty;
        float successReward = 0f;

        if (!hasReachedTarget)
        {
            // Before the first successful reach, preserve the original progress
            // shaping and ordinary proximity penalty.
            progressReward = progressWeight * (previousDistance - distance);
            proximityPenalty = -distancePenaltyWeight * distance;

            if (distance <= reachDistance)
            {
                hasReachedTarget = true;
                postReachStepsElapsed = 0;
                successReward = reachBonus; // Award only once, on first contact.

                Debug.Log(
                    $"[NicoAgentReach] Target reached. Continuing NN control for " +
                    $"{postReachSteps} additional decisions.");
            }
        }
        else
        {
            // During the post-reach phase, stop rewarding incremental approach.
            // Instead, penalize distance from the target on every decision while
            // the NN remains fully active. Movement and smoothness penalties also
            // continue, encouraging small/settled control commands.
            proximityPenalty = -postReachDistancePenaltyWeight * distance;
            postReachStepsElapsed++;
        }

        AddReward(
            progressReward + proximityPenalty + successReward +
            smoothnessPenalty + movementPenalty);

        previousDistance = distance;

        if (hasReachedTarget)
        {
            // A zero value preserves immediate termination after reaching.
            if (postReachSteps <= 0)
            {
                FinishEpisode(true);
                return;
            }

            // The target remains at the same sampled position. The hand continues
            // receiving NN actions for the configured number of additional decisions.
            if (postReachStepsElapsed >= postReachSteps)
            {
                // Count this episode as successful only if the hand is still within
                // the reach tolerance when the post-reach phase finishes.
                bool heldNearTarget = distance <= reachDistance;
                FinishEpisode(heldNearTarget);
                return;
            }
        }
        else
        {
            // Stall detection applies only before the first successful reach.
            // It must not interrupt the configured post-reach continuation phase.
            if (distance < bestDistance - noProgressEpsilon)
            {
                bestDistance = distance;
                noProgressSteps = 0;
            }
            else
            {
                noProgressSteps++;
            }

            if (noProgressSteps >= noProgressStepsToEndEpisode)
            {
                FinishEpisode(false);
                return;
            }
        }

        // maxEpisodeSteps is the maximum time allowed to reach the target.
        // Once the target is reached, the full post-reach interval is allowed
        // even if the reach happened near the pre-reach timeout.
        if (!hasReachedTarget && episodeSteps >= maxEpisodeSteps)
        {
            FinishEpisode(false);
            return;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Neutral actions for Heuristic/manual mode. Training and inference use the policy.
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        for (int i = 0; i < continuousActions.Length; i++)
            continuousActions[i] = 0f;
    }

    // ---------------------------------------------------------------------
    // AREA-BASED CURRICULUM PROGRESSION
    // ---------------------------------------------------------------------

    private void FinishEpisode(bool success)
    {
        RecordEpisodeResult(success);
        EndEpisode();
    }

    private void RecordEpisodeResult(bool success)
    {
        if (!enableCurriculumProgression)
            return;

        int windowSize = Mathf.Max(1, successWindowSize);

        recentEpisodeResults.Enqueue(success);
        while (recentEpisodeResults.Count > windowSize)
            recentEpisodeResults.Dequeue();

        if (recentEpisodeResults.Count < windowSize ||
            unlockedCurriculumStageCount >= curriculumRegionFractions.Count)
            return;

        int successes = recentEpisodeResults.Count(result => result);
        float successRate = successes / (float)recentEpisodeResults.Count;

        if (successRate >= successRateToUnlockNextRegion)
        {
            int oldStageCount = unlockedCurriculumStageCount;
            unlockedCurriculumStageCount = Mathf.Min(
                unlockedCurriculumStageCount + 1,
                curriculumRegionFractions.Count);
            recentEpisodeResults.Clear();

            Vector3 newlyUnlockedFraction = curriculumRegionFractions[unlockedCurriculumStageCount - 1];
            Debug.Log(
                $"[Reach Curriculum] Success rate {successRate:P0} over the last {windowSize} episodes. " +
                $"Unlocked region {unlockedCurriculumStageCount}/{curriculumRegionFractions.Count} " +
                $"(was {oldStageCount}/{curriculumRegionFractions.Count}); " +
                $"region fraction={newlyUnlockedFraction}. ");
        }
    }

    // ---------------------------------------------------------------------
    // SCENE VIEW GIZMOS
    // ---------------------------------------------------------------------

    private void OnDrawGizmosSelected()
    {
        if (targetSpawnVolume == null)
            return;

        BoxCollider volumeCollider = targetSpawnVolume.GetComponent<BoxCollider>();
        if (volumeCollider == null)
            return;

        Matrix4x4 previousMatrix = Gizmos.matrix;
        Color previousColor = Gizmos.color;

        Gizmos.matrix = targetSpawnVolume.transform.localToWorldMatrix;

        // Full spawn volume.
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(volumeCollider.center, volumeCollider.size);

        // Currently active curriculum sub-region. Once initialized, this outline
        // uses the safe target-pivot range (it accounts for the target cube size).
        if (curriculumRegionFractions != null && curriculumRegionFractions.Count > 0)
        {
            int stageIndex = Application.isPlaying
                ? Mathf.Clamp(unlockedCurriculumStageCount - 1, 0, curriculumRegionFractions.Count - 1)
                : 0;

            Vector3 fraction = curriculumRegionFractions[stageIndex];
            Gizmos.color = Color.yellow;

            if (spawnVolumeGeometryReady)
            {
                Vector3 regionHalfRange = Vector3.Scale(validTargetPivotHalfRangeLocal, fraction);
                Gizmos.DrawWireCube(validTargetPivotCenterLocal, regionHalfRange * 2f);
            }
            else
            {
                Gizmos.DrawWireCube(
                    volumeCollider.center,
                    Vector3.Scale(volumeCollider.size, fraction));
            }
        }

        Gizmos.matrix = previousMatrix;
        Gizmos.color = previousColor;
    }
}
