using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Validates the target spawn volume, calculates the safe target-pivot range
/// (accounting for target collider/renderer bounds), and samples reproducible
/// world positions inside a selected curriculum region. It can also precompute
/// a deterministic stratified target set for checkpoint evaluation.
/// </summary>
public sealed class TargetSpawnManager
{
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

        public int RangeInt(int minInclusive, int maxExclusive)
        {
            if (maxExclusive <= minInclusive)
                return minInclusive;

            uint range = (uint)(maxExclusive - minInclusive);
            return minInclusive + (int)(NextUInt() % range);
        }
    }

    private GameObject target;
    private GameObject targetSpawnVolume;
    private DeterministicRandom random;

    public BoxCollider SpawnVolumeCollider { get; private set; }
    public Vector3 ValidTargetPivotMinLocal { get; private set; }
    public Vector3 ValidTargetPivotMaxLocal { get; private set; }
    public Vector3 ValidTargetPivotCenterLocal { get; private set; }
    public Vector3 ValidTargetPivotHalfRangeLocal { get; private set; }
    public bool SpawnVolumeGeometryReady { get; private set; }

    public bool Initialize(
        GameObject targetObject,
        GameObject spawnVolumeObject,
        float boundaryMarginLocal,
        int seed)
    {
        target = targetObject;
        targetSpawnVolume = spawnVolumeObject;
        random = new DeterministicRandom(seed);

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

        SpawnVolumeCollider = targetSpawnVolume.GetComponent<BoxCollider>();
        if (SpawnVolumeCollider == null)
        {
            Debug.LogError("NicoAgentReach: TargetSpawnVolume must have a BoxCollider on the same GameObject.");
            return false;
        }

        if (SpawnVolumeCollider.size.x <= 0f ||
            SpawnVolumeCollider.size.y <= 0f ||
            SpawnVolumeCollider.size.z <= 0f)
        {
            Debug.LogError("NicoAgentReach: TargetSpawnVolume BoxCollider must have positive size on all axes.");
            return false;
        }

        CacheTargetColliderOffsetsAndCalculateValidRange(boundaryMarginLocal);

        if (ValidTargetPivotMinLocal.x > ValidTargetPivotMaxLocal.x ||
            ValidTargetPivotMinLocal.y > ValidTargetPivotMaxLocal.y ||
            ValidTargetPivotMinLocal.z > ValidTargetPivotMaxLocal.z)
        {
            Debug.LogError(
                "NicoAgentReach: the target cube is too large to fit inside TargetSpawnVolume " +
                "with the current Spawn Boundary Margin. Enlarge the volume or reduce the margin.");
            return false;
        }

        SpawnVolumeGeometryReady = true;
        return true;
    }

    public Vector3 GenerateRandomTargetWorldPosition(Vector3 curriculumRegionFraction)
    {
        Vector3 regionHalfRange = Vector3.Scale(
            ValidTargetPivotHalfRangeLocal,
            curriculumRegionFraction);

        Vector3 localTargetPivot = ValidTargetPivotCenterLocal + new Vector3(
            random.Range(-regionHalfRange.x, regionHalfRange.x),
            random.Range(-regionHalfRange.y, regionHalfRange.y),
            random.Range(-regionHalfRange.z, regionHalfRange.z));

        return targetSpawnVolume.transform.TransformPoint(localTargetPivot);
    }

    /// <summary>
    /// Precomputes a deterministic, ordered Latin-hypercube-style target set.
    /// For each local axis of the spawn volume,
    /// each target occupies a different equal-width stratum. The same seed,
    /// count, region fraction and spawn-volume geometry reproduce the same set.
    /// </summary>
    public List<Vector3> GenerateStratifiedTargetSet(
        Vector3 regionFraction,
        int targetCount,
        int seed)
    {
        int count = Mathf.Max(1, targetCount);
        Vector3 fraction = new Vector3(
            Mathf.Clamp01(regionFraction.x),
            Mathf.Clamp01(regionFraction.y),
            Mathf.Clamp01(regionFraction.z));
        Vector3 regionHalfRange = Vector3.Scale(ValidTargetPivotHalfRangeLocal, fraction);

        DeterministicRandom setRandom = new DeterministicRandom(seed);
        int[] xStrata = CreateAndShuffleStrata(count, setRandom);
        int[] yStrata = CreateAndShuffleStrata(count, setRandom);
        int[] zStrata = CreateAndShuffleStrata(count, setRandom);

        List<Vector3> positions = new List<Vector3>(count);
        for (int i = 0; i < count; i++)
        {
            float unitX = (xStrata[i] + setRandom.Range(0f, 1f)) / count;
            float unitY = (yStrata[i] + setRandom.Range(0f, 1f)) / count;
            float unitZ = (zStrata[i] + setRandom.Range(0f, 1f)) / count;

            Vector3 centeredUnitPosition = new Vector3(
                unitX * 2f - 1f,
                unitY * 2f - 1f,
                unitZ * 2f - 1f);

            Vector3 localTargetPivot = ValidTargetPivotCenterLocal +
                Vector3.Scale(regionHalfRange, centeredUnitPosition);
            positions.Add(targetSpawnVolume.transform.TransformPoint(localTargetPivot));
        }

        return positions;
    }

    private static int[] CreateAndShuffleStrata(int count, DeterministicRandom randomSource)
    {
        int[] strata = new int[count];
        for (int i = 0; i < count; i++)
            strata[i] = i;

        for (int i = count - 1; i > 0; i--)
        {
            int swapIndex = randomSource.RangeInt(0, i + 1);
            int temporary = strata[i];
            strata[i] = strata[swapIndex];
            strata[swapIndex] = temporary;
        }

        return strata;
    }

    private void CacheTargetColliderOffsetsAndCalculateValidRange(float boundaryMarginLocal)
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

                    EncapsulateOffset(
                        offsetInSpawnLocal,
                        ref offsetMin,
                        ref offsetMax,
                        ref foundAnyBounds);
                }
            }
            else if (targetCollider.enabled)
            {
                // Conservative fallback for non-box collider shapes: use world AABB corners.
                AddWorldBoundsCorners(
                    targetCollider.bounds,
                    ref offsetMin,
                    ref offsetMax,
                    ref foundAnyBounds);
            }
        }

        // If the target has no Collider components, use Renderer bounds so the visible
        // target still fits inside the volume. This is a conservative world-AABB approximation.
        if (!foundAnyBounds)
        {
            Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
            foreach (Renderer targetRenderer in renderers)
            {
                if (targetRenderer != null && targetRenderer.enabled)
                {
                    AddWorldBoundsCorners(
                        targetRenderer.bounds,
                        ref offsetMin,
                        ref offsetMax,
                        ref foundAnyBounds);
                }
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

        Vector3 volumeMin = SpawnVolumeCollider.center - SpawnVolumeCollider.size * 0.5f;
        Vector3 volumeMax = SpawnVolumeCollider.center + SpawnVolumeCollider.size * 0.5f;
        Vector3 margin = Vector3.one * boundaryMarginLocal;
        volumeMin += margin;
        volumeMax -= margin;

        // A target pivot is valid only if every target-collider corner remains inside the volume.
        ValidTargetPivotMinLocal = volumeMin - offsetMin;
        ValidTargetPivotMaxLocal = volumeMax - offsetMax;
        ValidTargetPivotCenterLocal = (ValidTargetPivotMinLocal + ValidTargetPivotMaxLocal) * 0.5f;
        ValidTargetPivotHalfRangeLocal = (ValidTargetPivotMaxLocal - ValidTargetPivotMinLocal) * 0.5f;
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

            EncapsulateOffset(
                offsetInSpawnLocal,
                ref offsetMin,
                ref offsetMax,
                ref foundAnyBounds);
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
}
