using Unity.MLAgents;

/// <summary>
/// Centralizes TensorBoard/ML-Agents statistics reporting. Metrics are reported
/// once per episode and never contribute to AddReward().
/// </summary>
public static class EvaluationMetricsReporter
{
    public static void ReportEpisode(
        StableHoldMetrics stableHold,
        ReachingSmoothnessMetrics smoothness)
    {
        StatsRecorder recorder = Academy.Instance.StatsRecorder;

        // Reaching smoothness availability diagnostics. The smoothness values
        // themselves are only submitted when their calculations are valid.
        recorder.Add(
            "Smoothness/ReachSegmentReached",
            smoothness.ReachedTarget ? 1f : 0f,
            StatAggregationMethod.Average);
        recorder.Add(
            "Smoothness/CartesianDimensionlessJerkValid",
            smoothness.DimensionlessJerkValid ? 1f : 0f,
            StatAggregationMethod.Average);
        recorder.Add(
            "Smoothness/SPARCValid",
            smoothness.SparcValid ? 1f : 0f,
            StatAggregationMethod.Average);
        recorder.Add(
            "Smoothness/ReachTrajectorySampleCount",
            smoothness.RawSampleCount,
            StatAggregationMethod.Average);

        if (smoothness.SamplingFrequencyHz > 0f)
        {
            recorder.Add(
                "Smoothness/PhysicsSamplingFrequencyHz",
                smoothness.SamplingFrequencyHz,
                StatAggregationMethod.Average);
        }

        if (smoothness.DimensionlessJerkValid)
        {
            recorder.Add(
                "Smoothness/CartesianDimensionlessSquaredJerk",
                smoothness.CartesianDimensionlessSquaredJerk,
                StatAggregationMethod.Average);
        }

        if (smoothness.SparcValid)
        {
            recorder.Add(
                "Smoothness/SPARC",
                smoothness.Sparc,
                StatAggregationMethod.Average);
        }

        if (smoothness.DimensionlessJerkValid || smoothness.SparcValid)
        {
            recorder.Add(
                "Smoothness/ReachTrajectoryDurationSeconds",
                smoothness.DurationSeconds,
                StatAggregationMethod.Average);
            recorder.Add(
                "Smoothness/ReachTrajectoryAmplitudeMeters",
                smoothness.AmplitudeMeters,
                StatAggregationMethod.Average);
            recorder.Add(
                "Smoothness/ReachTrajectoryAnalysisSampleCount",
                smoothness.AnalysisSampleCount,
                StatAggregationMethod.Average);
        }

        // One 0/1 value per episode; Average produces the stable-hold success rate.
        recorder.Add(
            "Evaluation/StableHoldSuccessRate",
            stableHold.Success ? 1f : 0f,
            StatAggregationMethod.Average);

        // Preserve the requested failure sentinel (1) for the combined settling-time curve.
        recorder.Add(
            "Evaluation/SettlingTimeSteps",
            stableHold.Success ? stableHold.SettlingTimeSteps : 1f,
            StatAggregationMethod.Average);

        // Companion statistic: average settling time across successful episodes only.
        if (stableHold.Success)
        {
            recorder.Add(
                "Evaluation/SettlingTimeSuccessfulEpisodes",
                stableHold.SettlingTimeSteps,
                StatAggregationMethod.Average);
        }

        recorder.Add(
            "Evaluation/HoldDistanceSampleCount",
            stableHold.HoldDistanceSampleCount,
            StatAggregationMethod.Average);

        if (stableHold.HoldDistanceStdDevValid)
        {
            recorder.Add(
                "Evaluation/HoldDistanceStdDev",
                stableHold.HoldDistanceStdDev,
                StatAggregationMethod.Average);
        }
    }
}
