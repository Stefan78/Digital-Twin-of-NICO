using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class NicoAgentReach : Agent
{
    private ArticulationBody nico_art_body;

    [Header("NICO parts to control")]
    [SerializeField] private bool useLeftHand = true;
    [SerializeField] private bool useRightHand = true;
    [SerializeField] private bool useHead = true;

    [Header("Collected Joints")]
    public List<ArticulationBody> leftHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> rightHandJoints = new List<ArticulationBody>();
    public List<ArticulationBody> headJoints = new List<ArticulationBody>();

    private List<ArticulationBody> activeJoints = new List<ArticulationBody>();

    private readonly Dictionary<ArticulationBody, float> defaultJointTargets =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, float> currentJointChanges =
        new Dictionary<ArticulationBody, float>();
    private readonly Dictionary<ArticulationBody, ArticulationBody[]> fingerChains =
        new Dictionary<ArticulationBody, ArticulationBody[]>();

    [Header("Curriculum Target Cubes")]
    [Tooltip("Place the target cubes in the scene at the desired world positions. The first N cubes are enabled by the curriculum.")]
    [SerializeField] private List<GameObject> targetCubes = new List<GameObject>();

    [Tooltip("Current goal cube. Assigned automatically from Target Cubes each episode. If Target Cubes is empty, this object is used as a fallback.")]
    public GameObject target;

    [Tooltip("End effector for the right hand")]
    public GameObject effector_1;

    [Tooltip("End effector for the left hand")]
    public GameObject effector_2;

    [SerializeField, Min(1)] private int initialTargetsUnlocked = 1;
    [SerializeField, Min(1)] private int successWindowSize = 50;
    [SerializeField, Range(0f, 1f)] private float successRateToUnlockNextTarget = 0.8f;

    private int unlockedTargetCount;
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();

    [Header("Episode Length")]
    [Tooltip("Maximum number of agent decisions in one episode. A timeout counts as a failed reaching episode.")]
    [SerializeField, Min(1)] private int maxEpisodeSteps = 1000;
    private int episodeSteps;

    [Header("Reach Reward Settings")]
    [SerializeField] private float reachDistance = 0.05f;
    [SerializeField] private float progressWeight = 3f;
    [SerializeField] private float distancePenaltyWeight = 0.02f;
    [SerializeField] private float reachBonus = 1f;
    [SerializeField] private float smoothnessWeight = 0.01f;
    [SerializeField] private float movementWeight = 0.005f;

    [Header("Stall Detection")]
    [SerializeField, Min(1)] private int noProgressStepsToLock = 100;
    [SerializeField, Min(0f)] private float noProgressEpsilon = 0.001f;

    private bool rightHandLocked;
    private bool leftHandLocked;
    private bool rightHandReached;
    private bool leftHandReached;

    private float previousRightDistance;
    private float previousLeftDistance;
    private float bestRightDistance;
    private float bestLeftDistance;
    private int rightNoProgressSteps;
    private int leftNoProgressSteps;

    private void CacheFingerChain(ArticulationBody root)
    {
        if (root == null)
            return;

        ArticulationBody[] joints = root.GetComponentsInChildren<ArticulationBody>();
        fingerChains[root] = joints.Where(j => j != root).ToArray();
    }

    private void CollectJointsInformation(ArticulationBody root)
    {
        foreach (ArticulationBody joint in root.GetComponentsInChildren<ArticulationBody>())
        {
            string jointName = joint.name.ToLowerInvariant();
            string jointTag = joint.tag.ToLowerInvariant();

            // Cache finger bases so one action can also drive their child joints.
            if (jointTag.Contains("base"))
                CacheFingerChain(joint);

            if (jointName.StartsWith("l_") && !jointTag.Contains("continuation"))
                leftHandJoints.Add(joint);
            else if (jointName.StartsWith("r_") && !jointTag.Contains("continuation"))
                rightHandJoints.Add(joint);
            else if (jointName.StartsWith("h_"))
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
        if (nico_art_body == null)
        {
            Debug.LogError("NicoAgentReach requires an ArticulationBody on the same GameObject as the Agent.");
            enabled = false;
            return;
        }

        // Clean up empty or duplicate Inspector references.
        targetCubes = targetCubes.Where(cube => cube != null).Distinct().ToList();

        // Backwards compatibility: if no list was assigned, use the original Target object.
        if (targetCubes.Count == 0 && target != null)
            targetCubes.Add(target);

        if (targetCubes.Count == 0)
        {
            Debug.LogError("NicoAgentReach: assign at least one cube to Target Cubes (or assign Target as a fallback).");
            enabled = false;
            return;
        }

        if (effector_1 == null || effector_2 == null)
        {
            Debug.LogError("NicoAgentReach: assign both end effectors (effector_1 and effector_2) in the Inspector.");
            enabled = false;
            return;
        }

        unlockedTargetCount = Mathf.Clamp(initialTargetsUnlocked, 1, targetCubes.Count);
        UpdateTargetCubeVisibility();
        target = targetCubes[0];

        CollectJointsInformation(nico_art_body);
        UpdateActiveJoints();

        foreach (ArticulationBody joint in activeJoints)
        {
            defaultJointTargets[joint] = joint.xDrive.target;
            currentJointChanges[joint] = 0f;
        }
    }

    private void UpdateTargetCubeVisibility()
    {
        // Stage 1 exposes one target. Later stages enable the first 2, 3, ... targets.
        for (int i = 0; i < targetCubes.Count; i++)
        {
            if (targetCubes[i] != null)
                targetCubes[i].SetActive(i < unlockedTargetCount);
        }
    }

    public void SetJointTarget(ArticulationBody joint, float angle)
    {
        ArticulationDrive drive = joint.xDrive;
        drive.target = angle;
        joint.xDrive = drive;

        if (fingerChains.TryGetValue(joint, out ArticulationBody[] chain))
        {
            foreach (ArticulationBody subJoint in chain)
            {
                float ratio = subJoint.tag.ToLowerInvariant().Contains("thumb") ? 0.25f : 1f;
                SetJointTarget(subJoint, angle * ratio);
            }
        }
    }

    public void ResetJoint(ArticulationBody joint, float jointDefault)
    {
        ArticulationDrive drive = joint.xDrive;
        drive.target = jointDefault;
        joint.xDrive = drive;

        joint.jointPosition = new ArticulationReducedSpace(jointDefault);
        joint.jointVelocity = new ArticulationReducedSpace(0.0f);
        joint.linearVelocity = Vector3.zero;
        joint.angularVelocity = Vector3.zero;

        if (fingerChains.TryGetValue(joint, out ArticulationBody[] chain))
        {
            foreach (ArticulationBody subJoint in chain)
                ResetJoint(subJoint, jointDefault);
        }
    }

    public override void OnEpisodeBegin()
    {
        episodeSteps = 0;

        rightHandLocked = false;
        leftHandLocked = false;
        rightHandReached = false;
        leftHandReached = false;
        rightNoProgressSteps = 0;
        leftNoProgressSteps = 0;

        UpdateTargetCubeVisibility();

        // Choose one goal uniformly from the currently unlocked target cubes.
        int selectedIndex = Random.Range(0, unlockedTargetCount);
        target = targetCubes[selectedIndex];

        // Reset joint positions, velocities, targets and accumulated action changes.
        foreach (ArticulationBody joint in activeJoints)
        {
            ResetJoint(joint, defaultJointTargets[joint]);
            currentJointChanges[joint] = 0f;
        }

        previousRightDistance = Vector3.Distance(
            effector_1.transform.position, target.transform.position);
        previousLeftDistance = Vector3.Distance(
            effector_2.transform.position, target.transform.position);

        bestRightDistance = previousRightDistance;
        bestLeftDistance = previousLeftDistance;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (ArticulationBody joint in activeJoints)
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

        // The policy receives the relative position from each hand to the selected goal cube.
        sensor.AddObservation(target.transform.position - effector_1.transform.position);
        sensor.AddObservation(target.transform.position - effector_2.transform.position);

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
        episodeSteps++;

        float maxChange = 2f;
        float changeMagnitude = 0.3f;
        float smoothnessSum = 0f;
        float movementSum = 0f;
        int controlledJointCount = 0;

        // Apply actions only to joints that have not been locked.
        foreach (var (joint, action) in activeJoints.Zip(
            actions.ContinuousActions, (joint, action) => (joint, action)))
        {
            if (IsJointLocked(joint))
            {
                currentJointChanges[joint] = 0f;
                continue;
            }

            float previousChange = currentJointChanges[joint];
            currentJointChanges[joint] = Mathf.Clamp(
                previousChange + action * changeMagnitude,
                -maxChange,
                maxChange);

            float newTarget = Mathf.Clamp(
                joint.xDrive.target + currentJointChanges[joint],
                joint.xDrive.lowerLimit,
                joint.xDrive.upperLimit);

            SetJointTarget(joint, newTarget);

            smoothnessSum += Mathf.Abs(currentJointChanges[joint] - previousChange) / maxChange;
            movementSum += Mathf.Abs(currentJointChanges[joint]) / maxChange;
            controlledJointCount++;
        }

        float normalizer = Mathf.Max(controlledJointCount, 1);
        float smoothnessPenalty = -smoothnessWeight * smoothnessSum / normalizer;
        float movementPenalty = -movementWeight * movementSum / normalizer;

        float rightDistance = Vector3.Distance(
            effector_1.transform.position, target.transform.position);
        float leftDistance = Vector3.Distance(
            effector_2.transform.position, target.transform.position);

        float successReward = 0f;

        // RIGHT HAND: lock after reaching the goal, or after a prolonged stall.
        if (!rightHandLocked)
        {
            if (rightDistance <= reachDistance)
            {
                rightHandReached = true;
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
                    rightHandLocked = true;
            }
        }

        // LEFT HAND: lock after reaching the goal, or after a prolonged stall.
        if (!leftHandLocked)
        {
            if (leftDistance <= reachDistance)
            {
                leftHandReached = true;
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
                    leftHandLocked = true;
            }
        }

        float progressReward = progressWeight *
            ((previousRightDistance - rightDistance) +
             (previousLeftDistance - leftDistance));

        float averageDistance = (rightDistance + leftDistance) / 2f;
        float proximityPenalty = -distancePenaltyWeight * averageDistance;

        float totalReward = progressReward + proximityPenalty + successReward
            + smoothnessPenalty + movementPenalty;

        AddReward(totalReward);

        previousRightDistance = rightDistance;
        previousLeftDistance = leftDistance;

        // An episode succeeds only when BOTH hands actually reached the selected cube.
        if (rightHandLocked && leftHandLocked)
        {
            FinishEpisode(rightHandReached && leftHandReached);
            return;
        }

        // Prevent a stalled episode from running indefinitely. Timeouts count as failures.
        if (episodeSteps >= maxEpisodeSteps)
        {
            FinishEpisode(false);
            return;
        }
    }

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

        // Evaluate only once the full recent-results window is available.
        if (recentEpisodeResults.Count < windowSize || unlockedTargetCount >= targetCubes.Count)
            return;

        int successes = recentEpisodeResults.Count(result => result);
        float successRate = successes / (float)recentEpisodeResults.Count;

        if (successRate >= successRateToUnlockNextTarget)
        {
            int oldTargetCount = unlockedTargetCount;
            unlockedTargetCount = Mathf.Min(unlockedTargetCount + 1, targetCubes.Count);

            // Start a fresh evaluation window for the harder curriculum stage.
            recentEpisodeResults.Clear();
            UpdateTargetCubeVisibility();

            Debug.Log(
                $"[Reach Curriculum] Success rate {successRate:P0} over the last {windowSize} episodes. " +
                $"Unlocked {unlockedTargetCount}/{targetCubes.Count} target cubes (was {oldTargetCount}).");
        }
    }
}
