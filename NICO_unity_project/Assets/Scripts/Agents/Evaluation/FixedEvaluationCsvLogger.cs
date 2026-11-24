using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

/// <summary>
/// Writes one raw row per fixed-set inference episode and a distribution summary
/// when the requested evaluation set has completed. This class does not affect
/// rewards, observations, actions, or episode termination.
/// </summary>
public sealed class FixedEvaluationCsvLogger
{
    private readonly string runLabel;
    private readonly int targetSeed;
    private readonly int requestedEpisodes;
    private string episodesCsvPath;
    private string summaryCsvPath;

    private readonly List<double> stableHoldSuccess = new List<double>();
    private readonly List<double> reachEvent = new List<double>();
    private readonly List<double> jerkValid = new List<double>();
    private readonly List<double> sparcValid = new List<double>();
    private readonly List<double> settlingAll = new List<double>();
    private readonly List<double> settlingSuccessfulOnly = new List<double>();
    private readonly List<double> holdDistanceStdDev = new List<double>();
    private readonly List<double> jerk = new List<double>();
    private readonly List<double> sparc = new List<double>();
    private readonly List<double> initialDistance = new List<double>();
    private readonly List<double> episodeSteps = new List<double>();
    private readonly List<double> cumulativeReward = new List<double>();

    private bool completed;

    public string EpisodesCsvPath => episodesCsvPath;
    public string SummaryCsvPath => summaryCsvPath;

    public FixedEvaluationCsvLogger(string requestedRunLabel, int seed, int requestedEpisodeCount)
    {
        runLabel = string.IsNullOrWhiteSpace(requestedRunLabel)
            ? "NICO_evaluation"
            : requestedRunLabel.Trim();
        targetSeed = seed;
        requestedEpisodes = Mathf.Max(1, requestedEpisodeCount);

        string safeLabel = SanitizeFileName(runLabel);
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string outputDirectory = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", "NicoEvaluationResults"));
        string fileStem = $"{safeLabel}_seed{targetSeed}_{timestamp}";

        episodesCsvPath = Path.Combine(outputDirectory, $"NicoEvalEpisodes_{fileStem}.csv");
        summaryCsvPath = Path.Combine(outputDirectory, $"NicoEvalSummary_{fileStem}.csv");

        try
        {
            Directory.CreateDirectory(outputDirectory);
            File.WriteAllText(
                episodesCsvPath,
                "RunLabel,Episode,TargetSeed,TargetWorldX,TargetWorldY,TargetWorldZ," +
                "InitialTargetDistanceM,DecisionSteps,LegacyReachAndHoldSuccess,CumulativeReward," +
                "StableHoldSuccess,SettlingTimeSteps,HoldDistanceStdDevValid,HoldDistanceStdDevM,HoldDistanceSampleCount," +
                "ReachedReachDistance,JerkValid,CartesianDimensionlessSquaredJerk,SPARCValid,SPARC," +
                "ReachDurationSeconds,ReachAmplitudeMeters,RawTrajectorySamples,AnalysisSamples,PhysicsSamplingFrequencyHz," +
                "FixedDeltaTime,TimeScale,PositionToleranceM,VelocityToleranceMPerS,RequiredHoldDecisions," +
                "ReachDistanceM,PostReachDecisions,MaxReachDecisions" + Environment.NewLine,
                Encoding.UTF8);

            Debug.Log("[NicoEvaluation] Per-episode CSV: " + episodesCsvPath);
            Debug.Log("[NicoEvaluation] Summary CSV (written after final episode): " + summaryCsvPath);
            Debug.Log("[NicoEvaluation] Results are saved in the Unity project folder: NicoEvaluationResults.");
        }
        catch (Exception exception)
        {
            Debug.LogError("[NicoEvaluation] Could not create evaluation CSV files: " + exception);
            episodesCsvPath = null;
            summaryCsvPath = null;
        }
    }

    public void WriteEpisode(
        int episodeNumber,
        Vector3 targetWorldPosition,
        float initialTargetDistance,
        int decisionSteps,
        bool legacyReachAndHoldSuccess,
        float episodeCumulativeReward,
        StableHoldMetrics stableHold,
        ReachingSmoothnessMetrics smoothness,
        float fixedDeltaTime,
        float timeScale,
        float positionTolerance,
        float velocityTolerance,
        int requiredHoldDecisions,
        float reachDistance,
        int postReachDecisions,
        int maxReachDecisions)
    {
        if (completed)
            return;

        stableHoldSuccess.Add(stableHold.Success ? 1.0 : 0.0);
        reachEvent.Add(smoothness.ReachedTarget ? 1.0 : 0.0);
        jerkValid.Add(smoothness.DimensionlessJerkValid ? 1.0 : 0.0);
        sparcValid.Add(smoothness.SparcValid ? 1.0 : 0.0);
        settlingAll.Add(stableHold.Success ? stableHold.SettlingTimeSteps : 1.0);
        if (stableHold.Success)
            settlingSuccessfulOnly.Add(stableHold.SettlingTimeSteps);
        if (stableHold.HoldDistanceStdDevValid)
            holdDistanceStdDev.Add(stableHold.HoldDistanceStdDev);
        if (smoothness.DimensionlessJerkValid)
            jerk.Add(smoothness.CartesianDimensionlessSquaredJerk);
        if (smoothness.SparcValid)
            sparc.Add(smoothness.Sparc);

        initialDistance.Add(initialTargetDistance);
        episodeSteps.Add(decisionSteps);
        cumulativeReward.Add(episodeCumulativeReward);

        if (string.IsNullOrEmpty(episodesCsvPath))
            return;

        string[] fields =
        {
            Csv(runLabel),
            I(episodeNumber),
            I(targetSeed),
            F(targetWorldPosition.x),
            F(targetWorldPosition.y),
            F(targetWorldPosition.z),
            F(initialTargetDistance),
            I(decisionSteps),
            B(legacyReachAndHoldSuccess),
            F(episodeCumulativeReward),
            B(stableHold.Success),
            I(stableHold.Success ? stableHold.SettlingTimeSteps : 1),
            B(stableHold.HoldDistanceStdDevValid),
            stableHold.HoldDistanceStdDevValid ? F(stableHold.HoldDistanceStdDev) : string.Empty,
            I(stableHold.HoldDistanceSampleCount),
            B(smoothness.ReachedTarget),
            B(smoothness.DimensionlessJerkValid),
            smoothness.DimensionlessJerkValid ? F(smoothness.CartesianDimensionlessSquaredJerk) : string.Empty,
            B(smoothness.SparcValid),
            smoothness.SparcValid ? F(smoothness.Sparc) : string.Empty,
            (smoothness.DimensionlessJerkValid || smoothness.SparcValid) ? F(smoothness.DurationSeconds) : string.Empty,
            (smoothness.DimensionlessJerkValid || smoothness.SparcValid) ? F(smoothness.AmplitudeMeters) : string.Empty,
            I(smoothness.RawSampleCount),
            I(smoothness.AnalysisSampleCount),
            F(smoothness.SamplingFrequencyHz),
            F(fixedDeltaTime),
            F(timeScale),
            F(positionTolerance),
            F(velocityTolerance),
            I(requiredHoldDecisions),
            F(reachDistance),
            I(postReachDecisions),
            I(maxReachDecisions)
        };

        try
        {
            File.AppendAllText(episodesCsvPath, string.Join(",", fields) + Environment.NewLine, Encoding.UTF8);
        }
        catch (Exception exception)
        {
            Debug.LogError("[NicoEvaluation] Failed to append episode CSV row: " + exception.Message);
        }
    }

    public void Complete(int completedEpisodes)
    {
        if (completed)
            return;

        completed = true;
        if (string.IsNullOrEmpty(summaryCsvPath))
        {
            Debug.LogWarning("[NicoEvaluation] Evaluation completed, but summary CSV could not be created.");
            return;
        }

        try
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendLine("RunLabel,TargetSeed,CompletedEpisodes,RequestedEpisodes");
            builder.AppendLine(string.Join(",", Csv(runLabel), I(targetSeed), I(completedEpisodes), I(requestedEpisodes)));
            builder.AppendLine();
            builder.AppendLine("Metric,ValidCount,Mean,Median,SampleStdDev,P25,P75,P90");

            AppendSummary(builder, "StableHoldSuccessRate", stableHoldSuccess);
            AppendSummary(builder, "ReachDistanceEventRate", reachEvent);
            AppendSummary(builder, "JerkValidRate", jerkValid);
            AppendSummary(builder, "SPARCValidRate", sparcValid);
            AppendSummary(builder, "SettlingTimeSteps_AllEpisodes_FailureSentinel1", settlingAll);
            AppendSummary(builder, "SettlingTimeSteps_SuccessfulEpisodesOnly", settlingSuccessfulOnly);
            AppendSummary(builder, "HoldDistanceStdDev_M", holdDistanceStdDev);
            AppendSummary(builder, "CartesianDimensionlessSquaredJerk", jerk);
            AppendSummary(builder, "SPARC", sparc);
            AppendSummary(builder, "InitialTargetDistance_M", initialDistance);
            AppendSummary(builder, "EpisodeDecisionSteps", episodeSteps);
            AppendSummary(builder, "EpisodeCumulativeReward_DiagnosticOnly", cumulativeReward);

            File.WriteAllText(summaryCsvPath, builder.ToString(), Encoding.UTF8);
            Debug.Log("[NicoEvaluation] Evaluation summary written: " + summaryCsvPath);
        }
        catch (Exception exception)
        {
            Debug.LogError("[NicoEvaluation] Failed to write evaluation summary: " + exception);
        }
    }

    private static void AppendSummary(StringBuilder builder, string metricName, List<double> values)
    {
        if (values == null || values.Count == 0)
        {
            builder.AppendLine(string.Join(",", Csv(metricName), "0", "", "", "", "", "", ""));
            return;
        }

        double[] sorted = values.OrderBy(value => value).ToArray();
        double mean = sorted.Average();
        double variance = 0.0;
        if (sorted.Length > 1)
        {
            for (int i = 0; i < sorted.Length; i++)
            {
                double difference = sorted[i] - mean;
                variance += difference * difference;
            }
            variance /= sorted.Length - 1; // Sample variance.
        }

        builder.AppendLine(string.Join(",",
            Csv(metricName),
            I(sorted.Length),
            D(mean),
            D(Percentile(sorted, 0.50)),
            D(Math.Sqrt(variance)),
            D(Percentile(sorted, 0.25)),
            D(Percentile(sorted, 0.75)),
            D(Percentile(sorted, 0.90))));
    }

    private static double Percentile(double[] sortedValues, double percentile)
    {
        if (sortedValues == null || sortedValues.Length == 0)
            return double.NaN;
        if (sortedValues.Length == 1)
            return sortedValues[0];

        double position = Mathf.Clamp01((float)percentile) * (sortedValues.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        double interpolation = position - lower;
        return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * interpolation;
    }

    private static string SanitizeFileName(string value)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        StringBuilder result = new StringBuilder(value.Length);
        foreach (char character in value)
        {
            if (Array.IndexOf(invalid, character) >= 0 || char.IsWhiteSpace(character))
                result.Append('_');
            else
                result.Append(character);
        }
        return result.ToString();
    }

    private static string Csv(string value)
    {
        if (value == null)
            return string.Empty;
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string B(bool value) => value ? "1" : "0";
    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
    private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
    private static string D(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}
