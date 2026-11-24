using System;
using UnityEngine;

/// <summary>
/// Evaluates the independent stable-hold definition and accumulates distance
/// variation statistics. It does not modify rewards or end episodes.
/// </summary>
public sealed class StableHoldEvaluator
{
    private readonly GameObject rightHandEffector;
    private readonly GameObject target;
    private readonly float positionTolerance;
    private readonly float velocityTolerance;
    private readonly int requiredHoldSteps;

    private ArticulationBody effectorVelocityArticulation;
    private ArticulationBody targetVelocityArticulation;
    private Rigidbody effectorVelocityRigidbody;
    private Rigidbody targetVelocityRigidbody;

    private Vector3 previousEvaluationRelativePosition;
    private float previousEvaluationSampleTime;
    private bool hasPreviousEvaluationSample;

    private int consecutiveStableEvaluationSteps;
    private bool stableHoldEvaluationSucceeded;
    private int stableHoldCompletionStep;

    // Welford's algorithm for population standard deviation of distance samples.
    private bool holdDistanceSamplingStarted;
    private int holdDistanceSampleCount;
    private double holdDistanceMean;
    private double holdDistanceM2;

    public StableHoldEvaluator(
        GameObject endEffector,
        GameObject targetObject,
        float positionTolerance,
        float velocityTolerance,
        int requiredHoldSteps)
    {
        rightHandEffector = endEffector;
        target = targetObject;
        this.positionTolerance = positionTolerance;
        this.velocityTolerance = velocityTolerance;
        this.requiredHoldSteps = Mathf.Max(1, requiredHoldSteps);

        CacheEvaluationVelocityBodies();
    }

    private void CacheEvaluationVelocityBodies()
    {
        effectorVelocityArticulation = rightHandEffector.GetComponentInParent<ArticulationBody>();
        targetVelocityArticulation = target.GetComponentInParent<ArticulationBody>();

        if (effectorVelocityArticulation == null)
            effectorVelocityRigidbody = rightHandEffector.GetComponentInParent<Rigidbody>();

        if (targetVelocityArticulation == null)
            targetVelocityRigidbody = target.GetComponentInParent<Rigidbody>();

        if (effectorVelocityArticulation == null && effectorVelocityRigidbody == null)
        {
            Debug.LogWarning(
                "[NicoAgentReach] No ArticulationBody/Rigidbody was found for the end effector. " +
                "Relative speed will be estimated from position changes between Agent decisions.");
        }

        // A stationary target normally has no physics body; its velocity is treated as zero.
    }

    public void Reset()
    {
        consecutiveStableEvaluationSteps = 0;
        stableHoldEvaluationSucceeded = false;
        stableHoldCompletionStep = 0;

        holdDistanceSamplingStarted = false;
        holdDistanceSampleCount = 0;
        holdDistanceMean = 0.0;
        holdDistanceM2 = 0.0;

        previousEvaluationRelativePosition =
            target.transform.position - rightHandEffector.transform.position;
        previousEvaluationSampleTime = Time.fixedTime;
        hasPreviousEvaluationSample = true;
    }

    public float CalculateRelativeEffectorSpeed()
    {
        Vector3 currentRelativePosition =
            target.transform.position - rightHandEffector.transform.position;

        bool hasPhysicsVelocitySource =
            effectorVelocityArticulation != null ||
            targetVelocityArticulation != null ||
            effectorVelocityRigidbody != null ||
            targetVelocityRigidbody != null;

        float relativeSpeed;
        if (hasPhysicsVelocitySource)
        {
            Vector3 effectorVelocity = GetPhysicsPointVelocity(
                rightHandEffector.transform,
                effectorVelocityArticulation,
                effectorVelocityRigidbody);
            Vector3 targetVelocity = GetPhysicsPointVelocity(
                target.transform,
                targetVelocityArticulation,
                targetVelocityRigidbody);

            relativeSpeed = (targetVelocity - effectorVelocity).magnitude;
        }
        else if (hasPreviousEvaluationSample)
        {
            // Fallback for transform-driven objects without physics bodies.
            float elapsedTime = Time.fixedTime - previousEvaluationSampleTime;
            relativeSpeed = elapsedTime > 0.000001f
                ? (currentRelativePosition - previousEvaluationRelativePosition).magnitude / elapsedTime
                : 0f;
        }
        else
        {
            relativeSpeed = 0f;
        }

        previousEvaluationRelativePosition = currentRelativePosition;
        previousEvaluationSampleTime = Time.fixedTime;
        hasPreviousEvaluationSample = true;
        return relativeSpeed;
    }

    private static Vector3 GetPhysicsPointVelocity(
        Transform pointTransform,
        ArticulationBody articulationBody,
        Rigidbody rigidbody)
    {
        if (articulationBody != null)
            return articulationBody.GetPointVelocity(pointTransform.position);

        if (rigidbody != null)
            return rigidbody.GetPointVelocity(pointTransform.position);

        return Vector3.zero;
    }

    public void Update(float distance, float relativeSpeed, int episodeDecisionStep)
    {
        // Hold-distance sampling starts at first entry into the positional tolerance
        // and continues until episode termination, matching the existing metric.
        if (!holdDistanceSamplingStarted && distance <= positionTolerance)
            holdDistanceSamplingStarted = true;

        if (holdDistanceSamplingStarted)
            AddHoldDistanceSample(distance);

        // Existential success: once a complete stable interval occurs, later departures
        // do not undo success for this episode.
        if (stableHoldEvaluationSucceeded)
            return;

        if (distance <= positionTolerance && relativeSpeed <= velocityTolerance)
            consecutiveStableEvaluationSteps++;
        else
            consecutiveStableEvaluationSteps = 0;

        if (consecutiveStableEvaluationSteps >= requiredHoldSteps)
        {
            stableHoldEvaluationSucceeded = true;
            stableHoldCompletionStep = episodeDecisionStep;

            Debug.Log(
                $"[NicoAgentReach] Stable-hold evaluation succeeded at decision {stableHoldCompletionStep}. " +
                $"Required consecutive decisions={requiredHoldSteps}, " +
                $"position tolerance={positionTolerance:F4}, " +
                $"relative-speed tolerance={velocityTolerance:F4} m/s.");
        }
    }

    private void AddHoldDistanceSample(float distance)
    {
        holdDistanceSampleCount++;
        double delta = distance - holdDistanceMean;
        holdDistanceMean += delta / holdDistanceSampleCount;
        double deltaAfterMeanUpdate = distance - holdDistanceMean;
        holdDistanceM2 += delta * deltaAfterMeanUpdate;
    }

    public StableHoldMetrics GetMetrics()
    {
        float standardDeviation = 0f;
        bool standardDeviationValid = holdDistanceSampleCount > 0;

        if (standardDeviationValid)
        {
            double populationVariance = holdDistanceM2 / holdDistanceSampleCount;
            standardDeviation = (float)Math.Sqrt(Math.Max(0.0, populationVariance));
        }

        return new StableHoldMetrics(
            stableHoldEvaluationSucceeded,
            stableHoldCompletionStep,
            holdDistanceSampleCount,
            standardDeviationValid,
            standardDeviation);
    }
}

public struct StableHoldMetrics
{
    public bool Success { get; }
    public int SettlingTimeSteps { get; }
    public int HoldDistanceSampleCount { get; }
    public bool HoldDistanceStdDevValid { get; }
    public float HoldDistanceStdDev { get; }

    public StableHoldMetrics(
        bool success,
        int settlingTimeSteps,
        int holdDistanceSampleCount,
        bool holdDistanceStdDevValid,
        float holdDistanceStdDev)
    {
        Success = success;
        SettlingTimeSteps = settlingTimeSteps;
        HoldDistanceSampleCount = holdDistanceSampleCount;
        HoldDistanceStdDevValid = holdDistanceStdDevValid;
        HoldDistanceStdDev = holdDistanceStdDev;
    }
}
