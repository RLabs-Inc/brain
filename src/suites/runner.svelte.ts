/**
 * Experiment Runner - Orchestrates Creature + World + Data Collection
 *
 * This is the simulation loop that runs experiments:
 * 1. World provides stimuli to creature
 * 2. Creature thinks (neural computation)
 * 3. Creature acts in world
 * 4. World updates physics
 * 5. World provides reward signal
 * 6. Data is collected
 *
 * Following sveltui pattern:
 * - $state for reactive properties (for UI observation)
 * - Async/await for GPU operations
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Creature, CreatureState } from '../creatures/types.ts'
import type { World, WorldState } from '../worlds/types.ts'
import type {
  Experiment,
  ExperimentResult,
  DataPoint,
  TimeSeries,
  Analysis,
  Suite,
  SuiteResult,
  Comparison,
  RunnerCallbacks
} from './types.ts'

// ============================================================================
// RUNNER STATE (for UI observation)
// ============================================================================

// Current experiment state (reactive for UI)
// Wrapped in object so we can mutate properties without reassigning the export
export const runnerState = $state({
  currentExperiment: null as string | null,
  currentStep: 0,
  totalSteps: 0,
  isRunning: false,
})

// ============================================================================
// DATA COLLECTION
// ============================================================================

/**
 * Collect a single data point during experiment.
 */
async function collectDataPoint(
  timestep: number,
  creature: Creature,
  world: World,
  cumulativeReward: number
): Promise<DataPoint> {
  const creatureState = await creature.getState()
  const worldState = world.getState()
  const reward = world.getReward(creature)

  return {
    timestep,
    creatureState,
    worldState,
    reward,
    cumulativeReward: cumulativeReward + reward
  }
}

/**
 * Convert data points to time series for analysis.
 */
function extractTimeSeries(data: DataPoint[]): Map<string, TimeSeries> {
  const series = new Map<string, TimeSeries>()

  // Extract reward time series
  const rewardValues = data.map(d => d.reward)
  const rewardTimestamps = data.map(d => d.timestep)
  series.set('reward', { name: 'reward', values: rewardValues, timestamps: rewardTimestamps })

  // Extract cumulative reward
  const cumRewardValues = data.map(d => d.cumulativeReward)
  series.set('cumulative_reward', {
    name: 'cumulative_reward',
    values: cumRewardValues,
    timestamps: rewardTimestamps
  })

  // Extract spike counts
  const spikeValues = data.map(d => d.creatureState.totalSpikes)
  series.set('spikes', { name: 'spikes', values: spikeValues, timestamps: rewardTimestamps })

  return series
}

// ============================================================================
// EXPERIMENT RUNNER
// ============================================================================

export interface RunOptions {
  dt?: number              // Timestep (default: 1.0 ms)
  recordEveryN?: number    // Record data every N steps (default: 1)
  seed?: number            // Random seed for reproducibility
}

/**
 * Run a single experiment and collect data.
 *
 * This is the core simulation loop:
 * sense → think → act → world.step → reward → collect
 */
export async function runExperiment(
  creature: Creature,
  world: World,
  timesteps: number,
  options: RunOptions = {},
  onStep?: (step: number, creature: CreatureState, world: WorldState) => void
): Promise<ExperimentResult> {
  const dt = options.dt ?? 1.0
  const recordEveryN = options.recordEveryN ?? 1
  const seed = options.seed ?? Date.now()

  // Initialize state
  runnerState.currentStep = 0
  runnerState.totalSteps = timesteps
  runnerState.isRunning = true

  const data: DataPoint[] = []
  let cumulativeReward = 0
  const startTime = new Date()

  try {
    for (let t = 0; t < timesteps; t++) {
      runnerState.currentStep = t

      // 1. World → Creature (sensory input)
      const stimuli = world.getStimuli(creature)
      creature.sense(stimuli)

      // 2. Creature thinks (neural computation)
      await creature.think(dt)

      // 3. Creature → World (motor output)
      const actions = creature.act()
      world.applyActions(creature, actions)

      // 4. World updates (physics, spawns, etc.)
      world.step(dt)

      // 5. Reward signal (for learning)
      const reward = world.getReward(creature)
      creature.setReward(reward)
      cumulativeReward += reward

      // 6. Collect data (optionally skip some steps for performance)
      if (t % recordEveryN === 0) {
        const dataPoint = await collectDataPoint(t, creature, world, cumulativeReward)
        data.push(dataPoint)
      }

      // 7. Callback for visualization
      if (onStep) {
        const creatureState = await creature.getState()
        const worldState = world.getState()
        onStep(t, creatureState, worldState)
      }
    }
  } finally {
    runnerState.isRunning = false
  }

  const endTime = new Date()

  // Get final states
  const finalCreatureState = await creature.getState()
  const finalWorldState = world.getState()

  return {
    experimentName: runnerState.currentExperiment ?? 'unnamed',
    hypothesis: '',  // Filled in by Experiment wrapper
    startTime,
    endTime,
    totalTimesteps: timesteps,
    data,
    metrics: extractTimeSeries(data),
    finalCumulativeReward: cumulativeReward,
    finalCreatureState,
    finalWorldState,
    seed,
    config: { dt, recordEveryN }
  }
}

// ============================================================================
// ANALYSIS HELPERS
// ============================================================================

/**
 * Basic statistical analysis of experiment results.
 */
export function analyzeResult(result: ExperimentResult): Analysis {
  const rewards = result.data.map(d => d.reward)

  // Basic stats
  const meanReward = rewards.reduce((a, b) => a + b, 0) / rewards.length
  const variance = rewards.reduce((a, b) => a + (b - meanReward) ** 2, 0) / rewards.length
  const stdReward = Math.sqrt(variance)
  const maxReward = Math.max(...rewards)
  const minReward = Math.min(...rewards)

  // Trend analysis (simple linear regression)
  const n = rewards.length
  const sumX = (n * (n - 1)) / 2
  const sumY = rewards.reduce((a, b) => a + b, 0)
  const sumXY = rewards.reduce((a, b, i) => a + i * b, 0)
  const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)

  let rewardTrend: 'increasing' | 'decreasing' | 'stable' | 'fluctuating'
  if (Math.abs(slope) < 0.001) {
    rewardTrend = 'stable'
  } else if (slope > 0) {
    rewardTrend = 'increasing'
  } else {
    rewardTrend = 'decreasing'
  }

  // Check for fluctuation (high variance relative to mean)
  if (stdReward > Math.abs(meanReward) * 0.5) {
    rewardTrend = 'fluctuating'
  }

  // Learning detection: is reward improving over time?
  const firstHalf = rewards.slice(0, Math.floor(rewards.length / 2))
  const secondHalf = rewards.slice(Math.floor(rewards.length / 2))
  const firstMean = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length
  const secondMean = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length
  const learningDetected = secondMean > firstMean * 1.1  // 10% improvement

  return {
    meanReward,
    stdReward,
    maxReward,
    minReward,
    rewardTrend,
    learningDetected,
    custom: {
      totalReward: result.finalCumulativeReward,
      rewardSlope: slope
    },
    slapAnswers: new Map()  // Filled in by experiment
  }
}

/**
 * Compare two experiment results (baseline vs treatment).
 */
export function compareResults(
  baseline: ExperimentResult,
  treatment: ExperimentResult
): Comparison {
  const baselineReward = baseline.finalCumulativeReward
  const treatmentReward = treatment.finalCumulativeReward

  const rewardDifference = treatmentReward - baselineReward
  const percentImprovement = baselineReward !== 0
    ? ((treatmentReward - baselineReward) / Math.abs(baselineReward)) * 100
    : treatmentReward > 0 ? 100 : 0

  // Simple significance test: is improvement > 10%?
  const statisticalSignificance = Math.abs(percentImprovement) > 10

  // Generate conclusion
  let conclusion: string
  if (percentImprovement > 10) {
    conclusion = `Treatment improved over baseline by ${percentImprovement.toFixed(1)}%`
  } else if (percentImprovement < -10) {
    conclusion = `Treatment performed worse than baseline by ${Math.abs(percentImprovement).toFixed(1)}%`
  } else {
    conclusion = `No significant difference between treatment and baseline`
  }

  return {
    baseline,
    treatment,
    rewardDifference,
    percentImprovement,
    statisticalSignificance,
    conclusion
  }
}

// ============================================================================
// SUITE RUNNER
// ============================================================================

/**
 * Run a complete experiment suite.
 */
export async function runSuite(
  suite: Suite,
  timestepsPerExperiment: number,
  callbacks?: RunnerCallbacks
): Promise<SuiteResult> {
  const startTime = new Date()
  const results = new Map<string, ExperimentResult>()
  const analyses = new Map<string, Analysis>()

  callbacks?.onSuiteStart?.(suite)

  for (const experiment of suite.experiments) {
    runnerState.currentExperiment = experiment.name
    callbacks?.onExperimentStart?.(experiment)

    // Run experiment
    const result = await experiment.run(timestepsPerExperiment, callbacks?.onStep)
    result.hypothesis = experiment.hypothesis
    results.set(experiment.name, result)

    // Analyze
    const analysis = experiment.analyze(result)
    analyses.set(experiment.name, analysis)

    callbacks?.onExperimentEnd?.(experiment, result)

    // Clean up
    experiment.teardown()
  }

  runnerState.currentExperiment = null

  // Generate comparisons (each experiment vs first baseline)
  const comparisons: Comparison[] = []
  const experimentNames = Array.from(results.keys())
  if (experimentNames.length > 1) {
    const baseline = results.get(experimentNames[0])!
    for (let i = 1; i < experimentNames.length; i++) {
      const treatment = results.get(experimentNames[i])!
      comparisons.push(compareResults(baseline, treatment))
    }
  }

  // Determine if hypothesis is supported
  // This is a simplified version - real science needs more rigor
  const lastExperiment = experimentNames[experimentNames.length - 1]
  const lastAnalysis = analyses.get(lastExperiment)
  const hypothesisSupported = lastAnalysis?.learningDetected ?? false

  const endTime = new Date()

  // Apply The Slap to the whole suite
  const suiteSlap = generateSuiteSlap(results, analyses, comparisons)

  callbacks?.onSuiteEnd?.(suite, {
    suiteName: suite.name,
    hypothesis: suite.hypothesis,
    startTime,
    endTime,
    results,
    analyses,
    comparisons,
    hypothesisSupported: hypothesisSupported ? true : 'inconclusive',
    conclusion: hypothesisSupported
      ? `Evidence supports hypothesis: ${suite.hypothesis}`
      : `Insufficient evidence for hypothesis: ${suite.hypothesis}`,
    suiteSlap
  })

  return {
    suiteName: suite.name,
    hypothesis: suite.hypothesis,
    startTime,
    endTime,
    results,
    analyses,
    comparisons,
    hypothesisSupported: hypothesisSupported ? true : 'inconclusive',
    conclusion: hypothesisSupported
      ? `Evidence supports hypothesis: ${suite.hypothesis}`
      : `Insufficient evidence for hypothesis: ${suite.hypothesis}`,
    suiteSlap
  }
}

/**
 * Generate The Slap questions for the entire suite.
 */
function generateSuiteSlap(
  results: Map<string, ExperimentResult>,
  analyses: Map<string, Analysis>,
  comparisons: Comparison[]
): string {
  const lines: string[] = [
    '=== THE SLAP: Honesty Protocol ===',
    '',
    '1. Did we hard-code any behavior that looks like learning?',
    '   Review: Check creature factory for any reward-seeking bias in initial weights.',
    '',
    '2. Did we design the test knowing the answer?',
    '   Review: Were experimental conditions chosen to guarantee success?',
    '',
    '3. Is this actually learning, or just our wiring?',
    '   Review: Compare innate-only vs learning-enabled results.',
    ''
  ]

  // Add comparison summary
  if (comparisons.length > 0) {
    lines.push('   Comparisons:')
    for (const comp of comparisons) {
      lines.push(`   - ${comp.baseline.experimentName} vs ${comp.treatment.experimentName}: ${comp.percentImprovement.toFixed(1)}% change`)
    }
    lines.push('')
  }

  lines.push(
    '4. Could random weights do this by chance?',
    '   Review: Run baseline with random wiring, compare to innate.',
    '',
    '5. Would this survive our harshest scrutiny?',
    '   Review: Show results to a skeptic. What would they criticize?',
    '',
    '=== END SLAP ==='
  )

  return lines.join('\n')
}
