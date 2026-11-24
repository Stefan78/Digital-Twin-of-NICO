using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Owns validation and progression of the area-based target-spawn curriculum.
/// The supplied region list is the same serialized list held by the Agent, so
/// validated values remain visible in the Inspector while the application runs.
/// </summary>
public sealed class CurriculumManager
{
    private List<Vector3> regionFractions;
    private readonly Queue<bool> recentEpisodeResults = new Queue<bool>();

    public int UnlockedStageCount { get; private set; }

    public Vector3 CurrentRegionFraction
    {
        get
        {
            int stageIndex = Mathf.Clamp(
                UnlockedStageCount - 1,
                0,
                regionFractions.Count - 1);
            return regionFractions[stageIndex];
        }
    }

    public void Initialize(List<Vector3> serializedRegionFractions, int initialUnlockedStages)
    {
        regionFractions = serializedRegionFractions;
        ValidateStages();
        UnlockedStageCount = Mathf.Clamp(initialUnlockedStages, 1, regionFractions.Count);
    }

    private void ValidateStages()
    {
        if (regionFractions == null)
            regionFractions = new List<Vector3>();

        if (regionFractions.Count == 0)
            regionFractions.Add(Vector3.one);

        Vector3 previous = Vector3.zero;
        for (int i = 0; i < regionFractions.Count; i++)
        {
            Vector3 requested = regionFractions[i];
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

            regionFractions[i] = stage;
            previous = stage;
        }
    }

    public void RecordEpisodeResult(
        bool success,
        bool progressionEnabled,
        int requestedWindowSize,
        float requiredSuccessRate)
    {
        if (!progressionEnabled)
            return;

        int windowSize = Mathf.Max(1, requestedWindowSize);
        recentEpisodeResults.Enqueue(success);
        while (recentEpisodeResults.Count > windowSize)
            recentEpisodeResults.Dequeue();

        if (recentEpisodeResults.Count < windowSize ||
            UnlockedStageCount >= regionFractions.Count)
        {
            return;
        }

        int successes = 0;
        foreach (bool result in recentEpisodeResults)
        {
            if (result)
                successes++;
        }

        float successRate = successes / (float)recentEpisodeResults.Count;
        if (successRate >= requiredSuccessRate)
        {
            int oldStageCount = UnlockedStageCount;
            UnlockedStageCount = Mathf.Min(UnlockedStageCount + 1, regionFractions.Count);
            recentEpisodeResults.Clear();

            Vector3 newlyUnlockedFraction = regionFractions[UnlockedStageCount - 1];
            Debug.Log(
                $"[Reach Curriculum] Success rate {successRate:P0} over the last {windowSize} episodes. " +
                $"Unlocked region {UnlockedStageCount}/{regionFractions.Count} " +
                $"(was {oldStageCount}/{regionFractions.Count}); " +
                $"region fraction={newlyUnlockedFraction}. ");
        }
    }
}
