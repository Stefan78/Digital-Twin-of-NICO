using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// Main ML-Agents component for training the NICO right hand to reach a target.
///
/// This class owns the Unity Inspector configuration and coordinates the helper
/// modules. Joint control, target spawning, curriculum progression, and episode
/// evaluation are implemented in separate ordinary C# classes.
///
/// Keep this component on the same GameObject as the Agent's ArticulationBody.
/// The helper classes are constructed automatically; they are not scene components.
/// </summary>
public class NicoAgentReach2 : Agent
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

    // ---------------------------------------------------------------------
    // AREA-BASED CURRICULUM
    // ---------------------------------------------------------------------

    [Header("Area-Based Curriculum")]
    [Tooltip("Each entry defines the fraction of the FULL valid target-center range available at this curriculum stage, independently for X, Y and Z. Stages should grow or stay equal in every axis. Defaults: central 25%, 50%, 75%, then the full volume.")]
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

    // ---------------------------------------------------------------------
    // EPISODE SETTINGS
    // ---------------------------------------------------------------------

    [Header("Episode Length")]
    [Tooltip("Maximum number of Agent decisions allowed to reach the target. After reaching, the configured Post-Reach Steps are added as an extra continuation phase.")]
    [SerializeField, Min(1)] private int maxEpisodeSteps = 1000;

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
    // STABLE-HOLD EVALUATION (METRICS ONLY - DOES NOT AFFECT REWARDS)
    // ---------------------------------------------------------------------

    [Header("Stable-Hold Evaluation (Metrics Only)")]
    [Tooltip("Maximum end-effector-to-target distance for the independent stable-hold success metric, in Unity world units (normally meters). Suggested starting value: 0.05 m.")]
    [SerializeField, Min(0f)] private float evaluationPositionTolerance = 0.05f;

    [Tooltip("Maximum relative speed between the end effector and target for stable-hold success, in meters per second. Suggested starting value: 0.05 m/s.")]
    [SerializeField, Min(0f)] private float evaluationVelocityTolerance = 0.05f;

    [Tooltip("Number of consecutive Agent decisions for which both position and relative-speed tolerances must be satisfied. Suggested starting value: 20 decisions.")]
    [SerializeField, Min(1)] private int evaluationHoldSteps = 20;

    // ---------------------------------------------------------------------
    // REACHING SMOOTHNESS EVALUATION (METRICS ONLY)
    // ---------------------------------------------------------------------

    [Header("Reaching Smoothness Evaluation (Metrics Only)")]
    [Tooltip("Odd sample-window length for local-polynomial (Savitzky-Golay-style) velocity and jerk estimation. 9 samples is a practical starting point at 50 Hz. Must be at least 7 and will be rounded up to an odd number.")]
    [SerializeField, Min(7)] private int smoothnessDerivativeWindowSamples = 9;

    [Tooltip("Movement onset is detected when speed exceeds this fraction of the peak speed for the required consecutive samples. A short window before the detected onset is retained for derivative estimation.")]
    [SerializeField, Range(0.01f, 0.25f)] private float movementOnsetSpeedFraction = 0.05f;

    [Tooltip("Number of consecutive physics samples above the movement-onset speed threshold required to confirm that movement has started.")]
    [SerializeField, Min(1)] private int movementOnsetConfirmationSamples = 3;

    [Tooltip("SPARC maximum frequency cutoff in Hz. 20 Hz follows common published settings; it is automatically limited to the physics-sampling Nyquist frequency.")]
    [SerializeField, Min(0.1f)] private float sparcMaxCutoffFrequencyHz = 20f;

    [Tooltip("SPARC normalized spectrum amplitude threshold used to select its adaptive frequency cutoff. The commonly used published value is 0.05.")]
    [SerializeField, Range(0.001f, 0.5f)] private float sparcAmplitudeThreshold = 0.05f;

    // ---------------------------------------------------------------------
    // FIXED-SET INFERENCE EVALUATION
    // ---------------------------------------------------------------------

    [Header("Fixed-Set Inference Evaluation (Post-Training Only)")]
    [Tooltip("Enable only when evaluating a trained ONNX model. The Agent uses a deterministic, reproducible sequence of target positions, bypasses curriculum progression, writes per-episode CSV files, and stops after the requested number of episodes.")]
    [SerializeField] private bool enableFixedEvaluationMode = false;

    [Tooltip("Number of targets in the fixed evaluation set. Every checkpoint should use the same value and evaluation seed; the deterministic stratified set depends on both.")]
    [SerializeField, Min(1)] private int fixedEvaluationEpisodeCount = 100;

    [Tooltip("Seed for the fixed, stratified target set. Keep identical across all checkpoints to reproduce the same ordered target positions. Episode count must also remain identical.")]
    [SerializeField] private int fixedEvaluationSeed = 314159;

    [Tooltip("Fraction of the safe valid target-pivot range used for evaluation. (1,1,1) evaluates the complete valid spawn volume. Keep identical across checkpoints.")]
    [SerializeField] private Vector3 fixedEvaluationRegionFraction = Vector3.one;

    [Tooltip("Human-readable model/checkpoint label included in the CSV output, e.g. checkpoint_10M_seed1.")]
    [SerializeField] private string fixedEvaluationRunLabel = "checkpoint_10M";

    [Tooltip("Write one row per completed episode plus a summary CSV in the project-level NicoEvaluationResults folder.")]
    [SerializeField] private bool writeFixedEvaluationCsv = true;

    [Tooltip("When supported by the installed ML-Agents package, force deterministic continuous-action selection before the Agent policy is initialized.")]
    [SerializeField] private bool forceDeterministicInference = true;

    // ---------------------------------------------------------------------
    // RUNTIME COORDINATORS
    // ---------------------------------------------------------------------

    private JointController jointController;
    private TargetSpawnManager targetSpawnManager;
    private CurriculumManager curriculumManager;
    private StableHoldEvaluator stableHoldEvaluator;
    private ReachingSmoothnessEvaluator reachingSmoothnessEvaluator;

    private int episodeSteps;
    private bool hasReachedTarget;
    private int postReachStepsElapsed;
    private float previousDistance;
    private float bestDistance;
    private int noProgressSteps;
    private bool actionSizeWarningIssued;

    private FixedEvaluationCsvLogger fixedEvaluationCsvLogger;
    private List<Vector3> fixedEvaluationTargetPositions = new List<Vector3>();
    private int fixedEvaluationEpisodesCompleted;
    private Vector3 currentEvaluationTargetWorldPosition;
    private float currentEvaluationInitialDistance;
    private bool fixedEvaluationRunComplete;

    // ---------------------------------------------------------------------
    // UNITY / ML-AGENTS LIFECYCLE
    // ---------------------------------------------------------------------

    private void Awake()
    {
        // BehaviorParameters hides its deterministic-inference flag in some
        // ML-Agents versions. Set it before Agent.OnEnable/LazyInitialize creates
        // the policy, using reflection to remain compatible across package versions.
        if (enableFixedEvaluationMode && forceDeterministicInference)
            TryEnableDeterministicInferenceBeforePolicyCreation();
    }

    private void TryEnableDeterministicInferenceBeforePolicyCreation()
    {
        Component behaviorParameters = null;
        Component[] components = GetComponents<Component>();
        foreach (Component component in components)
        {
            if (component != null && component.GetType().Name == "BehaviorParameters")
            {
                behaviorParameters = component;
                break;
            }
        }

        if (behaviorParameters == null)
        {
            Debug.LogWarning("[NicoEvaluation] Behavior Parameters component not found. Deterministic inference could not be configured automatically.");
            return;
        }

        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        System.Type componentType = behaviorParameters.GetType();

        try
        {
            PropertyInfo modelProperty = componentType.GetProperty("Model", flags);
            if (modelProperty != null && modelProperty.GetValue(behaviorParameters, null) == null)
            {
                Debug.LogError("[NicoEvaluation] Behavior Parameters has no ONNX Model assigned. Assign the trained model before evaluation.");
            }

            PropertyInfo behaviorTypeProperty = componentType.GetProperty("BehaviorType", flags);
            if (behaviorTypeProperty != null)
            {
                object behaviorType = behaviorTypeProperty.GetValue(behaviorParameters, null);
                if (behaviorType != null && !string.Equals(behaviorType.ToString(), "InferenceOnly", StringComparison.OrdinalIgnoreCase))
                {
                    Debug.LogWarning(
                        "[NicoEvaluation] Behavior Type is '" + behaviorType + "'. Set it to 'Inference Only' for a post-training benchmark.");
                }
            }

            PropertyInfo deterministicProperty = componentType.GetProperty("DeterministicInference", flags);
            if (deterministicProperty != null &&
                deterministicProperty.PropertyType == typeof(bool) &&
                deterministicProperty.CanWrite)
            {
                deterministicProperty.SetValue(behaviorParameters, true, null);
                Debug.Log("[NicoEvaluation] Deterministic inference enabled through BehaviorParameters.DeterministicInference.");
                return;
            }

            FieldInfo deterministicField = componentType.GetField("m_DeterministicInference", flags);
            if (deterministicField != null && deterministicField.FieldType == typeof(bool))
            {
                deterministicField.SetValue(behaviorParameters, true);
                Debug.Log("[NicoEvaluation] Deterministic inference enabled through the serialized BehaviorParameters field.");
                return;
            }
        }
        catch (System.Exception exception)
        {
            Debug.LogWarning("[NicoEvaluation] Could not set deterministic inference automatically: " + exception.Message);
            return;
        }

        Debug.LogWarning(
            "[NicoEvaluation] This installed ML-Agents version does not expose a recognized deterministic-inference setting. " +
            "The evaluation target sequence is still deterministic, but action selection may not be. Check your ML-Agents version.");
    }

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

        targetSpawnManager = new TargetSpawnManager();
        int effectiveSpawnSeed = enableFixedEvaluationMode ? fixedEvaluationSeed : spawnPositionSeed;
        if (!targetSpawnManager.Initialize(target, targetSpawnVolume, spawnBoundaryMarginLocal, effectiveSpawnSeed))
        {
            enabled = false;
            return;
        }

        if (curriculumRegionFractions == null)
            curriculumRegionFractions = new List<Vector3>();

        curriculumManager = new CurriculumManager();
        curriculumManager.Initialize(curriculumRegionFractions, initialCurriculumStagesUnlocked);

        jointController = new JointController();
        if (!jointController.Initialize(
                transform,
                autoDetectRightHandJoints,
                rightHandJoints,
                jointControlSettings,
                out string jointConfigurationError))
        {
            Debug.LogError(jointConfigurationError);
            enabled = false;
            return;
        }

        // Keep the Inspector list synchronized with the detected/cleaned list,
        // matching the original auto-detection behaviour.
        rightHandJoints = jointController.RightHandJoints;

        if (rightHandJoints.Count == 0)
        {
            Debug.LogError("NicoAgentReach: no right-hand joints were found or assigned. Check the joint list or the r_ naming convention.");
            enabled = false;
            return;
        }

        if (jointController.NNControlledJoints.Count == 0)
        {
            Debug.LogError("NicoAgentReach: all right-hand joints are marked fixed. At least one joint must remain NN-controlled.");
            enabled = false;
            return;
        }

        stableHoldEvaluator = new StableHoldEvaluator(
            rightHandEffector,
            target,
            evaluationPositionTolerance,
            evaluationVelocityTolerance,
            evaluationHoldSteps);

        reachingSmoothnessEvaluator = new ReachingSmoothnessEvaluator(
            smoothnessDerivativeWindowSamples,
            movementOnsetSpeedFraction,
            movementOnsetConfirmationSamples,
            sparcMaxCutoffFrequencyHz,
            sparcAmplitudeThreshold);

        fixedEvaluationEpisodesCompleted = 0;
        fixedEvaluationRunComplete = false;
        if (enableFixedEvaluationMode)
        {
            fixedEvaluationRegionFraction = new Vector3(
                Mathf.Clamp01(fixedEvaluationRegionFraction.x),
                Mathf.Clamp01(fixedEvaluationRegionFraction.y),
                Mathf.Clamp01(fixedEvaluationRegionFraction.z));

            fixedEvaluationTargetPositions = targetSpawnManager.GenerateStratifiedTargetSet(
                fixedEvaluationRegionFraction,
                Mathf.Max(1, fixedEvaluationEpisodeCount),
                fixedEvaluationSeed);

            if (writeFixedEvaluationCsv)
            {
                fixedEvaluationCsvLogger = new FixedEvaluationCsvLogger(
                    fixedEvaluationRunLabel,
                    fixedEvaluationSeed,
                    Mathf.Max(1, fixedEvaluationEpisodeCount));
            }

            Debug.Log(
                "[NicoEvaluation] FIXED EVALUATION MODE enabled. " +
                $"Episodes={Mathf.Max(1, fixedEvaluationEpisodeCount)}, seed={fixedEvaluationSeed}, " +
                $"region fraction={fixedEvaluationRegionFraction}, target-set sampler=deterministic stratified sequence. " +
                "Curriculum progression is bypassed. Keep spawn-volume geometry and all reset settings identical across checkpoints.");
        }

        LogJointConfiguration();
        LogSpawnVolumeConfiguration();
    }

    private void FixedUpdate()
    {
        if (reachingSmoothnessEvaluator != null && rightHandEffector != null)
        {
            reachingSmoothnessEvaluator.SamplePhysicsStep(
                rightHandEffector.transform.position,
                Time.fixedTime,
                Time.fixedDeltaTime);
        }
    }

    public override void OnEpisodeBegin()
    {
        episodeSteps = 0;
        noProgressSteps = 0;
        actionSizeWarningIssued = false;
        hasReachedTarget = false;
        postReachStepsElapsed = 0;

        if (fixedEvaluationRunComplete)
            return;

        // Training uses the currently unlocked curriculum region. Fixed evaluation
        // bypasses curriculum and samples from the same full configured region on
        // every run, using the evaluation-specific seed initialized above.
        if (enableFixedEvaluationMode)
        {
            int targetIndex = Mathf.Clamp(
                fixedEvaluationEpisodesCompleted,
                0,
                fixedEvaluationTargetPositions.Count - 1);
            target.transform.position = fixedEvaluationTargetPositions[targetIndex];
        }
        else
        {
            target.transform.position = targetSpawnManager.GenerateRandomTargetWorldPosition(
                curriculumManager.CurrentRegionFraction);
        }

        // Reset configured fixed joints and NN-controlled joints to their episode poses.
        jointController.ResetEpisodeJoints();

        currentEvaluationTargetWorldPosition = target.transform.position;
        currentEvaluationInitialDistance = Vector3.Distance(
            rightHandEffector.transform.position,
            target.transform.position);

        // Initialize evaluation histories after all episode resets, so teleporting
        // the hand into its starting pose is not counted as movement.
        stableHoldEvaluator.Reset();
        reachingSmoothnessEvaluator.Begin(
            rightHandEffector.transform.position,
            Time.fixedTime,
            Time.fixedDeltaTime);

        previousDistance = Vector3.Distance(
            rightHandEffector.transform.position, target.transform.position);
        bestDistance = previousDistance;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Observe the CURRENT PHYSICAL coordinate of every right-hand joint.
        // This includes fixed joints and makes physics-induced deviations observable.
        foreach (ArticulationBody joint in jointController.RightHandJoints)
            sensor.AddObservation(jointController.NormalizeJointCoordinate(joint));

        // 2. Observe accumulated action-change commands for ALL right-hand joints.
        // Fixed joints have no NN command, so their accumulated change is zero.
        float changeScale = Mathf.Max(maxAccumulatedChange, 0.0001f);
        foreach (ArticulationBody joint in jointController.RightHandJoints)
        {
            jointController.CurrentJointChanges.TryGetValue(joint, out float accumulatedChange);
            sensor.AddObservation(Mathf.Clamp(accumulatedChange / changeScale, -1f, 1f));
        }

        // 3. Relative target position from the right-hand end effector (3 values).
        sensor.AddObservation(
            target.transform.position - rightHandEffector.transform.position);

        // 4. Optional phase indicator: 0 while approaching, 1 after first reaching.
        if (includePostReachPhaseObservation)
            sensor.AddObservation(hasReachedTarget ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        episodeSteps++;

        int availableActions = actions.ContinuousActions.Length;
        if (availableActions != jointController.NNControlledJoints.Count && !actionSizeWarningIssued)
        {
            Debug.LogError(
                $"NicoAgentReach: Behavior Parameters provides {availableActions} continuous actions, " +
                $"but {jointController.NNControlledJoints.Count} joints are marked for NN control. Set Continuous Actions " +
                $"to {jointController.NNControlledJoints.Count}.");
            actionSizeWarningIssued = true;
        }

        float smoothnessSum = 0f;
        float movementSum = 0f;
        int controlledJointCount = 0;
        int actionCount = Mathf.Min(availableActions, jointController.NNControlledJoints.Count);

        // The NN continues controlling every non-fixed joint both before and
        // after reaching the target. No joints are frozen in the post-reach phase.
        for (int i = 0; i < actionCount; i++)
        {
            ArticulationBody joint = jointController.NNControlledJoints[i];
            float action = Mathf.Clamp(actions.ContinuousActions[i], -1f, 1f);

            // Incremental controller: NN output changes a bounded target-step command.
            float previousChange = jointController.CurrentJointChanges[joint];
            jointController.CurrentJointChanges[joint] = Mathf.Clamp(
                previousChange + action * actionChangeMagnitude,
                -maxAccumulatedChange,
                maxAccumulatedChange);

            float newTarget = Mathf.Clamp(
                joint.xDrive.target + jointController.CurrentJointChanges[joint],
                joint.xDrive.lowerLimit,
                joint.xDrive.upperLimit);

            jointController.SetJointTarget(joint, newTarget);

            float normalizer = Mathf.Max(maxAccumulatedChange, 0.0001f);
            smoothnessSum += Mathf.Abs(jointController.CurrentJointChanges[joint] - previousChange) / normalizer;
            movementSum += Mathf.Abs(jointController.CurrentJointChanges[joint]) / normalizer;
            controlledJointCount++;
        }

        float jointNormalizer = Mathf.Max(controlledJointCount, 1);
        float smoothnessPenalty = -smoothnessWeight * smoothnessSum / jointNormalizer;
        float movementPenalty = -movementWeight * movementSum / jointNormalizer;

        float distance = Vector3.Distance(
            rightHandEffector.transform.position, target.transform.position);
        float relativeSpeed = stableHoldEvaluator.CalculateRelativeEffectorSpeed();

        // Independent evaluation. It never contributes to AddReward().
        stableHoldEvaluator.Update(distance, relativeSpeed, episodeSteps);

        float progressReward = 0f;
        float proximityPenalty;
        float successReward = 0f;

        if (!hasReachedTarget)
        {
            progressReward = progressWeight * (previousDistance - distance);
            proximityPenalty = -distancePenaltyWeight * distance;

            if (distance <= reachDistance)
            {
                // Reaching smoothness ends at the first reach event. Stabilization
                // is evaluated separately by StableHoldEvaluator.
                reachingSmoothnessEvaluator.StopSampling(
                    true,
                    rightHandEffector.transform.position,
                    Time.fixedTime);
                hasReachedTarget = true;
                postReachStepsElapsed = 0;
                successReward = reachBonus;

                Debug.Log(
                    $"[NicoAgentReach] Target reached. Continuing NN control for " +
                    $"{postReachSteps} additional decisions.");
            }
        }
        else
        {
            proximityPenalty = -postReachDistancePenaltyWeight * distance;
            postReachStepsElapsed++;
        }

        // Reward calculation is unchanged by the evaluation modules.
        AddReward(
            progressReward + proximityPenalty + successReward +
            smoothnessPenalty + movementPenalty);

        previousDistance = distance;

        if (hasReachedTarget)
        {
            if (postReachSteps <= 0)
            {
                FinishEpisode(true);
                return;
            }

            if (postReachStepsElapsed >= postReachSteps)
            {
                bool heldNearTarget = distance <= reachDistance;
                FinishEpisode(heldNearTarget);
                return;
            }
        }
        else
        {
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
        // Once reached, the full post-reach interval is allowed even near timeout.
        if (!hasReachedTarget && episodeSteps >= maxEpisodeSteps)
        {
            FinishEpisode(false);
            return;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        for (int i = 0; i < continuousActions.Length; i++)
            continuousActions[i] = 0f;
    }

    // ---------------------------------------------------------------------
    // EPISODE COMPLETION AND REPORTING
    // ---------------------------------------------------------------------

    private void FinishEpisode(bool curriculumSuccess)
    {
        // Stop an incomplete reaching trajectory without marking it as reached.
        reachingSmoothnessEvaluator.StopSampling(
            false,
            rightHandEffector.transform.position,
            Time.fixedTime);

        ReachingSmoothnessMetrics smoothnessMetrics = reachingSmoothnessEvaluator.CalculateMetrics();
        StableHoldMetrics stableHoldMetrics = stableHoldEvaluator.GetMetrics();
        EvaluationMetricsReporter.ReportEpisode(stableHoldMetrics, smoothnessMetrics);

        if (enableFixedEvaluationMode)
        {
            int episodeNumber = fixedEvaluationEpisodesCompleted + 1;
            if (fixedEvaluationCsvLogger != null)
            {
                fixedEvaluationCsvLogger.WriteEpisode(
                    episodeNumber,
                    currentEvaluationTargetWorldPosition,
                    currentEvaluationInitialDistance,
                    episodeSteps,
                    curriculumSuccess,
                    GetCumulativeReward(),
                    stableHoldMetrics,
                    smoothnessMetrics,
                    Time.fixedDeltaTime,
                    Time.timeScale,
                    evaluationPositionTolerance,
                    evaluationVelocityTolerance,
                    evaluationHoldSteps,
                    reachDistance,
                    postReachSteps,
                    maxEpisodeSteps);
            }

            fixedEvaluationEpisodesCompleted++;
            Debug.Log(
                $"[NicoEvaluation] Completed episode {fixedEvaluationEpisodesCompleted}/" +
                $"{Mathf.Max(1, fixedEvaluationEpisodeCount)}. " +
                $"Stable-hold success={stableHoldMetrics.Success}, " +
                $"Jerk valid={smoothnessMetrics.DimensionlessJerkValid}, " +
                $"SPARC valid={smoothnessMetrics.SparcValid}.");

            // Do not call EndEpisode on the final test. EndEpisode immediately
            // invokes OnEpisodeBegin; disabling this Agent is the clean way to
            // finish the finite benchmark without starting an extra episode.
            if (fixedEvaluationEpisodesCompleted >= Mathf.Max(1, fixedEvaluationEpisodeCount))
            {
                fixedEvaluationRunComplete = true;
                if (fixedEvaluationCsvLogger != null)
                    fixedEvaluationCsvLogger.Complete(fixedEvaluationEpisodesCompleted);

                Debug.Log(
                    "[NicoEvaluation] Fixed evaluation completed. The Agent and its DecisionRequester are now disabled. " +
                    "See the CSV paths reported above for per-episode results and distribution summaries.");
                DecisionRequester decisionRequester = GetComponent<DecisionRequester>();
                if (decisionRequester != null)
                    decisionRequester.enabled = false;
                enabled = false;
                return;
            }
        }
        else
        {
            // Preserve curriculum progression only during training mode.
            curriculumManager.RecordEpisodeResult(
                curriculumSuccess,
                enableCurriculumProgression,
                successWindowSize,
                successRateToUnlockNextRegion);
        }

        EndEpisode();
    }

    // ---------------------------------------------------------------------
    // DIAGNOSTIC LOGGING
    // ---------------------------------------------------------------------

    private void LogJointConfiguration()
    {
        Debug.Log("[NicoAgentReach] Right-hand joint configuration:");

        foreach (ArticulationBody joint in jointController.RightHandJoints)
        {
            if (jointController.TryGetFixedTarget(joint, out float fixedTarget))
            {
                Debug.Log($"[NicoAgentReach] {joint.name}: FIXED drive target = {fixedTarget:F2}");
            }
            else
            {
                int actionIndex = jointController.NNControlledJoints.IndexOf(joint);
                Debug.Log($"[NicoAgentReach] {joint.name}: NN continuous action = {actionIndex}");
            }
        }

        int expectedObservations = jointController.RightHandJoints.Count * 2 + 3 +
            (includePostReachPhaseObservation ? 1 : 0);
        Debug.Log(
            $"[NicoAgentReach] Right-hand joints: {jointController.RightHandJoints.Count}; " +
            $"fixed: {jointController.FixedJointCount}; NN-controlled: {jointController.NNControlledJoints.Count}. " +
            $"Expected Behavior Parameters: {expectedObservations} vector observations, " +
            $"{jointController.NNControlledJoints.Count} continuous actions.");
    }

    private void LogSpawnVolumeConfiguration()
    {
        Debug.Log(
            "[NicoAgentReach] TargetSpawnVolume ready. " +
            $"Valid target-pivot range in volume-local coordinates: min={targetSpawnManager.ValidTargetPivotMinLocal}, " +
            $"max={targetSpawnManager.ValidTargetPivotMaxLocal}. " +
            $"Spawn volume BoxCollider enabled={targetSpawnManager.SpawnVolumeCollider.enabled} (it may remain disabled).");

        for (int i = 0; i < curriculumRegionFractions.Count; i++)
        {
            Debug.Log(
                $"[NicoAgentReach] Curriculum region {i + 1}/{curriculumRegionFractions.Count}: " +
                $"fraction={curriculumRegionFractions[i]}");
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

        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(volumeCollider.center, volumeCollider.size);

        if (curriculumRegionFractions != null && curriculumRegionFractions.Count > 0)
        {
            int stageIndex = Application.isPlaying && curriculumManager != null
                ? Mathf.Clamp(curriculumManager.UnlockedStageCount - 1, 0, curriculumRegionFractions.Count - 1)
                : 0;

            Vector3 fraction = curriculumRegionFractions[stageIndex];
            Gizmos.color = Color.yellow;

            if (targetSpawnManager != null && targetSpawnManager.SpawnVolumeGeometryReady)
            {
                Vector3 regionHalfRange = Vector3.Scale(
                    targetSpawnManager.ValidTargetPivotHalfRangeLocal,
                    fraction);
                Gizmos.DrawWireCube(
                    targetSpawnManager.ValidTargetPivotCenterLocal,
                    regionHalfRange * 2f);
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
