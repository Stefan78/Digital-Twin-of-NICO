using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Collects end-effector positions at physics-step frequency and calculates
/// Cartesian dimensionless squared jerk and SPARC over the reaching segment only.
/// The segment ends at the first reachDistance event; stabilization is excluded.
/// </summary>
public sealed class ReachingSmoothnessEvaluator
{
    private const int SmoothnessPolynomialOrder = 5;
    private const int SparcZeroPaddingLevel = 4;

    private readonly int derivativeWindowSamples;
    private readonly float movementOnsetSpeedFraction;
    private readonly int movementOnsetConfirmationSamples;
    private readonly float sparcMaxCutoffFrequencyHz;
    private readonly float sparcAmplitudeThreshold;

    private readonly List<Vector3> trajectorySamples = new List<Vector3>();
    private bool samplingActive;
    private bool reachedTarget;
    private float sampleInterval;
    private float lastSampleTime;
    private bool hasLastSampleTime;
    private Vector3 prePhysicsInitialPosition;
    private bool hasPrePhysicsInitialPosition;

    private ReachingSmoothnessMetrics cachedMetrics;
    private bool metricsCalculated;

    public int RawSampleCount => trajectorySamples.Count;

    public ReachingSmoothnessEvaluator(
        int derivativeWindowSamples,
        float movementOnsetSpeedFraction,
        int movementOnsetConfirmationSamples,
        float sparcMaxCutoffFrequencyHz,
        float sparcAmplitudeThreshold)
    {
        this.derivativeWindowSamples = Mathf.Max(7, derivativeWindowSamples);
        this.movementOnsetSpeedFraction = movementOnsetSpeedFraction;
        this.movementOnsetConfirmationSamples = Mathf.Max(1, movementOnsetConfirmationSamples);
        this.sparcMaxCutoffFrequencyHz = sparcMaxCutoffFrequencyHz;
        this.sparcAmplitudeThreshold = sparcAmplitudeThreshold;
    }

    public void Begin(Vector3 initialPosition, float currentTime, float fixedDeltaTime)
    {
        trajectorySamples.Clear();
        samplingActive = true;
        reachedTarget = false;
        sampleInterval = Mathf.Max(fixedDeltaTime, 0.000001f);
        lastSampleTime = currentTime;
        hasLastSampleTime = true;

        // OnEpisodeBegin resets ArticulationBody joint positions immediately, but
        // the end-effector Transform may not reflect the reset until the next
        // physics simulation. Do not use this potentially stale pre-physics pose
        // as the first trajectory sample. The first SamplePhysicsStep at a later
        // fixed time becomes the synchronized baseline instead.
        prePhysicsInitialPosition = initialPosition;
        hasPrePhysicsInitialPosition = true;

        cachedMetrics = ReachingSmoothnessMetrics.Empty;
        metricsCalculated = false;
    }

    /// <summary>Call from the Agent's FixedUpdate, once per physics step.</summary>
    public void SamplePhysicsStep(Vector3 position, float currentTime, float fixedDeltaTime)
    {
        if (!samplingActive)
            return;

        if (sampleInterval <= 0f)
            sampleInterval = Mathf.Max(fixedDeltaTime, 0.000001f);

        float minimumTimeAdvance = Mathf.Max(0.000001f, sampleInterval * 0.25f);
        if (hasLastSampleTime && currentTime - lastSampleTime < minimumTimeAdvance)
            return;

        if (trajectorySamples.Count == 0)
        {
            float elapsedSinceBegin = Mathf.Max(currentTime - lastSampleTime, sampleInterval);
            if (hasPrePhysicsInitialPosition)
            {
                float resetPoseDelta = Vector3.Distance(prePhysicsInitialPosition, position);
                float impliedTransitionSpeed = resetPoseDelta / elapsedSinceBegin;
                Debug.Log(
                    $"[SmoothnessDiagnostic] First physics-synchronized baseline captured. " +
                    $"prePhysicsToBaselineDelta={resetPoseDelta:F6} m, " +
                    $"elapsed={elapsedSinceBegin:F6} s, " +
                    $"impliedTransitionSpeed={impliedTransitionSpeed:F4} m/s. " +
                    "This transition is excluded from the reaching trajectory.");
                hasPrePhysicsInitialPosition = false;
            }

            // This is the first end-effector pose observed after the physics clock
            // has advanced beyond OnEpisodeBegin. It avoids differentiating across
            // a stale Transform pose from before the articulation reset was applied.
            trajectorySamples.Add(position);
            lastSampleTime = currentTime;
            hasLastSampleTime = true;
            return;
        }

        trajectorySamples.Add(position);
        lastSampleTime = currentTime;
        hasLastSampleTime = true;
    }

    public void StopSampling(bool didReachTarget, Vector3 currentPosition, float currentTime)
    {
        if (!samplingActive)
            return;

        if (didReachTarget)
        {
            float timeSinceLastSample = currentTime - lastSampleTime;
            float minimumTimeAdvance = Mathf.Max(0.000001f, sampleInterval * 0.25f);

            // Capture the reach-event pose if it belongs to a later physics tick;
            // otherwise replace the last sample to avoid a duplicate-time point.
            if (!hasLastSampleTime || timeSinceLastSample >= minimumTimeAdvance)
            {
                trajectorySamples.Add(currentPosition);
                lastSampleTime = currentTime;
                hasLastSampleTime = true;
            }
            else if (trajectorySamples.Count > 0)
            {
                trajectorySamples[trajectorySamples.Count - 1] = currentPosition;
            }

            reachedTarget = true;
        }

        samplingActive = false;
    }

    public ReachingSmoothnessMetrics CalculateMetrics()
    {
        if (metricsCalculated)
            return cachedMetrics;

        metricsCalculated = true;
        cachedMetrics = CalculateMetricsInternal();
        return cachedMetrics;
    }

    private ReachingSmoothnessMetrics CalculateMetricsInternal()
    {
        bool jerkValid = false;
        bool sparcValid = false;
        float dimensionlessSquaredJerk = 0f;
        float sparc = 0f;
        int analysisSampleCount = 0;
        float durationSeconds = 0f;
        float amplitudeMeters = 0f;

        if (!reachedTarget)
        {
            return new ReachingSmoothnessMetrics(
                false, false, 0f, 0f, trajectorySamples.Count,
                sampleInterval > 0f ? 1f / sampleInterval : 0f,
                0, 0f, 0f);
        }

        int rawSampleCount = trajectorySamples.Count;
        if (rawSampleCount < 7)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: TooFewRawSamples. " +
                $"rawSamples={rawSampleCount}, requiredMinimum=7.");
            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                sampleInterval > 0f ? 1f / sampleInterval : 0f,
                0, 0f, 0f);
        }

        int windowSize = Mathf.Max(7, derivativeWindowSamples);
        if ((windowSize & 1) == 0)
            windowSize++;
        if (windowSize > rawSampleCount)
        {
            windowSize = rawSampleCount;
            if ((windowSize & 1) == 0)
                windowSize--;
        }
        if (windowSize < 7)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: InvalidDerivativeWindow. " +
                $"rawSamples={rawSampleCount}, effectiveWindow={windowSize}.");
            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                sampleInterval > 0f ? 1f / sampleInterval : 0f,
                0, 0f, 0f);
        }

        float dt = Mathf.Max(sampleInterval, 0.000001f);
        Vector3[] smoothedPositions = new Vector3[rawSampleCount];
        Vector3[] velocities = new Vector3[rawSampleCount];
        Vector3[] jerks = new Vector3[rawSampleCount];
        float[] speeds = new float[rawSampleCount];

        // Local fifth-order polynomial fit over a centred odd window.
        // This is a Savitzky-Golay-style position/derivative estimator.
        for (int i = 0; i < rawSampleCount; i++)
        {
            if (!TryEstimatePositionVelocityAndJerk(
                trajectorySamples,
                i,
                windowSize,
                dt,
                out smoothedPositions[i],
                out velocities[i],
                out jerks[i]))
            {
                Debug.LogWarning(
                    $"[SmoothnessDiagnostic] REJECTED: DerivativeEstimationFailed. " +
                    $"failedAtSample={i}, rawSamples={rawSampleCount}, " +
                    $"windowSize={windowSize}, dt={dt:F6}s.");
                return new ReachingSmoothnessMetrics(
                    true, false, 0f, 0f, rawSampleCount,
                    1f / dt, 0, 0f, 0f);
            }

            speeds[i] = velocities[i].magnitude;
        }

        float peakSpeed = 0f;
        for (int i = 0; i < speeds.Length; i++)
            peakSpeed = Mathf.Max(peakSpeed, speeds[i]);

        if (peakSpeed <= 0.000001f)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: PeakSpeedTooLow. " +
                $"rawSamples={rawSampleCount}, peakSpeed={peakSpeed:E6} m/s, dt={dt:F6}s.");
            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                1f / dt, 0, 0f, 0f);
        }

        // Detect movement onset at a fraction of peak speed, requiring consecutive
        // samples. Retain one filter half-window before onset to keep the initial ramp.
        float onsetThreshold = peakSpeed * Mathf.Clamp(movementOnsetSpeedFraction, 0.001f, 0.5f);
        int confirmationSamples = Mathf.Max(1, movementOnsetConfirmationSamples);
        int detectedOnset = -1;

        for (int i = 0; i <= rawSampleCount - confirmationSamples; i++)
        {
            bool confirmed = true;
            for (int j = 0; j < confirmationSamples; j++)
            {
                if (speeds[i + j] < onsetThreshold)
                {
                    confirmed = false;
                    break;
                }
            }

            if (confirmed)
            {
                detectedOnset = i;
                break;
            }
        }

        if (detectedOnset < 0)
        {
            // Diagnostic: determine how long the longest run above the onset
            // threshold was. This reveals whether the consecutive-sample
            // requirement is rejecting otherwise valid reaching trajectories.
            int longestRun = 0;
            int currentRun = 0;
            int longestRunStart = -1;
            int currentRunStart = -1;
            int peakSpeedIndex = 0;

            for (int i = 0; i < speeds.Length; i++)
            {
                if (speeds[i] > speeds[peakSpeedIndex])
                    peakSpeedIndex = i;

                if (speeds[i] >= onsetThreshold)
                {
                    if (currentRun == 0)
                        currentRunStart = i;

                    currentRun++;
                    if (currentRun > longestRun)
                    {
                        longestRun = currentRun;
                        longestRunStart = currentRunStart;
                    }
                }
                else
                {
                    currentRun = 0;
                    currentRunStart = -1;
                }
            }

            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: MovementOnsetNotDetected. " +
                $"rawSamples={rawSampleCount}, peakSpeed={peakSpeed:F5} m/s " +
                $"(index {peakSpeedIndex}), onsetThreshold={onsetThreshold:F5} m/s " +
                $"({Mathf.Clamp(movementOnsetSpeedFraction, 0.001f, 0.5f):P1} of peak), " +
                $"requiredConsecutiveSamples={confirmationSamples}, " +
                $"longestRunAboveThreshold={longestRun} " +
                $"(starts at index {longestRunStart}). " +
                $"If longestRun is smaller than requiredConsecutiveSamples, the onset confirmation rule is the cause.");

            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                1f / dt, 0, 0f, 0f);
        }

        int filterHalfWindow = (windowSize - 1) / 2;
        int startIndex = Mathf.Max(0, detectedOnset - filterHalfWindow);
        int endIndex = rawSampleCount - 1;
        int segmentSampleCount = endIndex - startIndex + 1;

        // A short segment cannot support a credible third-derivative estimate.
        if (segmentSampleCount < windowSize || segmentSampleCount < 9)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: AnalysisSegmentTooShort. " +
                $"rawSamples={rawSampleCount}, onsetIndex={detectedOnset}, " +
                $"startIndex={startIndex}, analysisSamples={segmentSampleCount}, " +
                $"requiredAtLeast={Mathf.Max(windowSize, 9)}.");
            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                1f / dt, segmentSampleCount, 0f, 0f);
        }

        float duration = (endIndex - startIndex) * dt;
        float amplitude = Vector3.Distance(
            smoothedPositions[startIndex], smoothedPositions[endIndex]);

        // The amplitude normalization is undefined for a near-zero displacement.
        if (duration <= 0.000001f || amplitude <= 0.00001f)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] REJECTED: MovementAmplitudeOrDurationTooSmall. " +
                $"duration={duration:F6}s, amplitude={amplitude:F8}m, " +
                $"analysisSamples={segmentSampleCount}.");
            return new ReachingSmoothnessMetrics(
                true, false, 0f, 0f, rawSampleCount,
                1f / dt, segmentSampleCount, duration, amplitude);
        }

        analysisSampleCount = segmentSampleCount;
        durationSeconds = duration;
        amplitudeMeters = amplitude;

        // Hogan & Sternad's positive dimensionless squared-jerk cost, extended
        // to 3D by summing squared Cartesian jerk components.
        double integratedSquaredJerk = 0.0;
        for (int i = startIndex; i < endIndex; i++)
        {
            double jerkSquaredA = jerks[i].sqrMagnitude;
            double jerkSquaredB = jerks[i + 1].sqrMagnitude;
            integratedSquaredJerk += 0.5 * (jerkSquaredA + jerkSquaredB) * dt;
        }

        double durationToFifth = Math.Pow(duration, 5.0);
        double amplitudeSquared = (double)amplitude * amplitude;
        double dimensionlessJerk = durationToFifth * integratedSquaredJerk / amplitudeSquared;
        if (IsFinite(dimensionlessJerk) && dimensionlessJerk >= 0.0)
        {
            dimensionlessSquaredJerk = (float)dimensionlessJerk;
            jerkValid = IsFinite(dimensionlessSquaredJerk);
        }

        // SPARC uses the same movement segment, operating on speed rather than jerk.
        // It is calculated independently so one metric may remain valid if the other fails.
        float[] segmentSpeeds = new float[segmentSampleCount];
        Array.Copy(speeds, startIndex, segmentSpeeds, 0, segmentSampleCount);
        float sampleRateHz = 1f / dt;
        if (TryCalculateSparc(segmentSpeeds, sampleRateHz, out float sparcResult))
        {
            sparc = sparcResult;
            sparcValid = IsFinite(sparc);
        }

        if (!jerkValid)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] Jerk calculation returned an invalid value. " +
                $"duration={duration:F4}s, amplitude={amplitude:F5}m, " +
                $"analysisSamples={segmentSampleCount}, integratedSquaredJerk={integratedSquaredJerk:E6}.");
        }

        if (!sparcValid)
        {
            Debug.LogWarning(
                $"[SmoothnessDiagnostic] SPARC calculation failed. " +
                $"segmentSamples={segmentSampleCount}, sampleRate={sampleRateHz:F2}Hz, " +
                $"cutoff={sparcMaxCutoffFrequencyHz:F2}Hz, amplitudeThreshold={sparcAmplitudeThreshold:F3}.");
        }

        return new ReachingSmoothnessMetrics(
            reachedTarget,
            jerkValid,
            dimensionlessSquaredJerk,
            sparc,
            rawSampleCount,
            sampleRateHz,
            analysisSampleCount,
            durationSeconds,
            amplitudeMeters,
            sparcValid);
    }

    private bool TryEstimatePositionVelocityAndJerk(
        List<Vector3> positions,
        int centreIndex,
        int windowSize,
        float dt,
        out Vector3 smoothedPosition,
        out Vector3 velocity,
        out Vector3 jerk)
    {
        smoothedPosition = Vector3.zero;
        velocity = Vector3.zero;
        jerk = Vector3.zero;

        int polynomialSize = SmoothnessPolynomialOrder + 1;
        int halfWindow = (windowSize - 1) / 2;
        int firstIndex = Mathf.Clamp(
            centreIndex - halfWindow,
            0,
            positions.Count - windowSize);

        double[,] normalMatrix = new double[polynomialSize, polynomialSize];
        double[] rhsX = new double[polynomialSize];
        double[] rhsY = new double[polynomialSize];
        double[] rhsZ = new double[polynomialSize];

        for (int sample = firstIndex; sample < firstIndex + windowSize; sample++)
        {
            // Scale sample coordinates to keep the least-squares system well-conditioned.
            double normalizedTime = (sample - centreIndex) / (double)halfWindow;
            double[] basis = new double[polynomialSize];
            basis[0] = 1.0;
            for (int power = 1; power < polynomialSize; power++)
                basis[power] = basis[power - 1] * normalizedTime;

            Vector3 position = positions[sample];
            for (int row = 0; row < polynomialSize; row++)
            {
                rhsX[row] += basis[row] * position.x;
                rhsY[row] += basis[row] * position.y;
                rhsZ[row] += basis[row] * position.z;

                for (int column = 0; column < polynomialSize; column++)
                    normalMatrix[row, column] += basis[row] * basis[column];
            }
        }

        if (!TrySolveLinearSystem(normalMatrix, rhsX, out double[] coefficientsX) ||
            !TrySolveLinearSystem(normalMatrix, rhsY, out double[] coefficientsY) ||
            !TrySolveLinearSystem(normalMatrix, rhsZ, out double[] coefficientsZ))
        {
            return false;
        }

        double timeScale = halfWindow * (double)dt;
        if (timeScale <= 0.0000001)
            return false;

        double firstDerivativeScale = 1.0 / timeScale;
        double thirdDerivativeScale = 6.0 / Math.Pow(timeScale, 3.0);

        smoothedPosition = new Vector3(
            (float)coefficientsX[0],
            (float)coefficientsY[0],
            (float)coefficientsZ[0]);

        velocity = new Vector3(
            (float)(coefficientsX[1] * firstDerivativeScale),
            (float)(coefficientsY[1] * firstDerivativeScale),
            (float)(coefficientsZ[1] * firstDerivativeScale));

        jerk = new Vector3(
            (float)(coefficientsX[3] * thirdDerivativeScale),
            (float)(coefficientsY[3] * thirdDerivativeScale),
            (float)(coefficientsZ[3] * thirdDerivativeScale));

        return IsFinite(smoothedPosition) && IsFinite(velocity) && IsFinite(jerk);
    }

    private static bool TrySolveLinearSystem(
        double[,] matrix,
        double[] rightHandSide,
        out double[] solution)
    {
        int size = rightHandSide.Length;
        solution = new double[size];
        double[,] augmented = new double[size, size + 1];

        for (int row = 0; row < size; row++)
        {
            for (int column = 0; column < size; column++)
                augmented[row, column] = matrix[row, column];
            augmented[row, size] = rightHandSide[row];
        }

        // Gaussian elimination with partial pivoting.
        for (int pivot = 0; pivot < size; pivot++)
        {
            int bestRow = pivot;
            double bestMagnitude = Math.Abs(augmented[pivot, pivot]);
            for (int row = pivot + 1; row < size; row++)
            {
                double magnitude = Math.Abs(augmented[row, pivot]);
                if (magnitude > bestMagnitude)
                {
                    bestMagnitude = magnitude;
                    bestRow = row;
                }
            }

            if (bestMagnitude < 1e-12)
                return false;

            if (bestRow != pivot)
            {
                for (int column = pivot; column <= size; column++)
                {
                    double temporary = augmented[pivot, column];
                    augmented[pivot, column] = augmented[bestRow, column];
                    augmented[bestRow, column] = temporary;
                }
            }

            double pivotValue = augmented[pivot, pivot];
            for (int column = pivot; column <= size; column++)
                augmented[pivot, column] /= pivotValue;

            for (int row = 0; row < size; row++)
            {
                if (row == pivot)
                    continue;

                double factor = augmented[row, pivot];
                for (int column = pivot; column <= size; column++)
                    augmented[row, column] -= factor * augmented[pivot, column];
            }
        }

        for (int row = 0; row < size; row++)
        {
            solution[row] = augmented[row, size];
            if (!IsFinite(solution[row]))
                return false;
        }

        return true;
    }

    private bool TryCalculateSparc(float[] speedProfile, float sampleRateHz, out float sparcValue)
    {
        sparcValue = 0f;
        if (speedProfile == null || speedProfile.Length < 5 || sampleRateHz <= 0f)
            return false;

        int fftSize = 1;
        while (fftSize < speedProfile.Length)
            fftSize <<= 1;
        fftSize <<= SparcZeroPaddingLevel;

        // Guard against excessive allocation for unusually long trajectories.
        if (fftSize > (1 << 20))
            return false;

        double[] real = new double[fftSize];
        double[] imaginary = new double[fftSize];
        for (int i = 0; i < speedProfile.Length; i++)
            real[i] = speedProfile[i]; // Remaining values are zero-padding.

        ComputeRadixTwoFft(real, imaginary);

        int positiveFrequencyBinCount = fftSize / 2;
        double dcMagnitude = Math.Sqrt(real[0] * real[0] + imaginary[0] * imaginary[0]);
        if (dcMagnitude <= 1e-12)
            return false;

        double effectiveMaximumFrequency = Math.Min(
            Math.Max(0.1, sparcMaxCutoffFrequencyHz),
            sampleRateHz * 0.5);
        int maximumBin = Math.Min(
            positiveFrequencyBinCount,
            (int)Math.Floor(effectiveMaximumFrequency * fftSize / sampleRateHz));
        if (maximumBin < 2)
            return false;

        double[] normalizedMagnitude = new double[maximumBin + 1];
        for (int bin = 0; bin <= maximumBin; bin++)
        {
            double magnitude = Math.Sqrt(
                real[bin] * real[bin] + imaginary[bin] * imaginary[bin]);
            normalizedMagnitude[bin] = magnitude / dcMagnitude;
        }

        // Adaptive cutoff: last frequency bin up to fc whose normalized amplitude
        // reaches the specified threshold.
        double threshold = Math.Max(0.001, Math.Min(0.5, sparcAmplitudeThreshold));
        int cutoffBin = -1;
        for (int bin = 1; bin <= maximumBin; bin++)
        {
            if (normalizedMagnitude[bin] >= threshold)
                cutoffBin = bin;
        }

        if (cutoffBin < 1)
            return false;

        double normalizedFrequencyStep = 1.0 / cutoffBin;
        double arcLength = 0.0;
        for (int bin = 1; bin <= cutoffBin; bin++)
        {
            double magnitudeDifference =
                normalizedMagnitude[bin] - normalizedMagnitude[bin - 1];
            arcLength += Math.Sqrt(
                normalizedFrequencyStep * normalizedFrequencyStep +
                magnitudeDifference * magnitudeDifference);
        }

        double result = -arcLength;
        if (!IsFinite(result))
            return false;

        sparcValue = (float)result;
        return IsFinite(sparcValue);
    }

    private static void ComputeRadixTwoFft(double[] real, double[] imaginary)
    {
        int length = real.Length;
        if (length == 0 || (length & (length - 1)) != 0)
            throw new ArgumentException("FFT input length must be a non-zero power of two.");

        // Bit-reversal permutation.
        int j = 0;
        for (int i = 1; i < length; i++)
        {
            int bit = length >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if (i < j)
            {
                double temporaryReal = real[i];
                real[i] = real[j];
                real[j] = temporaryReal;

                double temporaryImaginary = imaginary[i];
                imaginary[i] = imaginary[j];
                imaginary[j] = temporaryImaginary;
            }
        }

        // Iterative Cooley-Tukey radix-2 FFT.
        for (int blockLength = 2; blockLength <= length; blockLength <<= 1)
        {
            double angle = -2.0 * Math.PI / blockLength;
            double stepReal = Math.Cos(angle);
            double stepImaginary = Math.Sin(angle);
            int halfBlock = blockLength >> 1;

            for (int blockStart = 0; blockStart < length; blockStart += blockLength)
            {
                double twiddleReal = 1.0;
                double twiddleImaginary = 0.0;

                for (int offset = 0; offset < halfBlock; offset++)
                {
                    int evenIndex = blockStart + offset;
                    int oddIndex = evenIndex + halfBlock;

                    double oddReal =
                        real[oddIndex] * twiddleReal - imaginary[oddIndex] * twiddleImaginary;
                    double oddImaginary =
                        real[oddIndex] * twiddleImaginary + imaginary[oddIndex] * twiddleReal;

                    double evenReal = real[evenIndex];
                    double evenImaginary = imaginary[evenIndex];

                    real[evenIndex] = evenReal + oddReal;
                    imaginary[evenIndex] = evenImaginary + oddImaginary;
                    real[oddIndex] = evenReal - oddReal;
                    imaginary[oddIndex] = evenImaginary - oddImaginary;

                    double nextTwiddleReal =
                        twiddleReal * stepReal - twiddleImaginary * stepImaginary;
                    double nextTwiddleImaginary =
                        twiddleReal * stepImaginary + twiddleImaginary * stepReal;
                    twiddleReal = nextTwiddleReal;
                    twiddleImaginary = nextTwiddleImaginary;
                }
            }

            if (blockLength == length)
                break; // Prevent integer overflow in the loop increment.
        }
    }

    private static bool IsFinite(double value)
    {
        return !double.IsNaN(value) && !double.IsInfinity(value);
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool IsFinite(Vector3 value)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
    }
}

public struct ReachingSmoothnessMetrics
{
    public bool ReachedTarget { get; }
    public bool DimensionlessJerkValid { get; }
    public float CartesianDimensionlessSquaredJerk { get; }
    public float Sparc { get; }
    public bool SparcValid { get; }
    public int RawSampleCount { get; }
    public float SamplingFrequencyHz { get; }
    public int AnalysisSampleCount { get; }
    public float DurationSeconds { get; }
    public float AmplitudeMeters { get; }

    public static ReachingSmoothnessMetrics Empty => new ReachingSmoothnessMetrics(
        false, false, 0f, 0f, 0, 0f, 0, 0f, 0f, false);

    public ReachingSmoothnessMetrics(
        bool reachedTarget,
        bool dimensionlessJerkValid,
        float dimensionlessSquaredJerk,
        float sparc,
        int rawSampleCount,
        float samplingFrequencyHz,
        int analysisSampleCount,
        float durationSeconds,
        float amplitudeMeters,
        bool sparcValid = false)
    {
        ReachedTarget = reachedTarget;
        DimensionlessJerkValid = dimensionlessJerkValid;
        CartesianDimensionlessSquaredJerk = dimensionlessSquaredJerk;
        Sparc = sparc;
        SparcValid = sparcValid;
        RawSampleCount = rawSampleCount;
        SamplingFrequencyHz = samplingFrequencyHz;
        AnalysisSampleCount = analysisSampleCount;
        DurationSeconds = durationSeconds;
        AmplitudeMeters = amplitudeMeters;
    }
}
