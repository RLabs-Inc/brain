/**
 * Experiment Suite Types - Scientific Testing Framework
 *
 * Experiments test hypotheses about learning and behavior.
 * Suites group related experiments that test one big hypothesis.
 *
 * THE HONESTY PROTOCOL (The Slap) is built into this interface.
 * Every experiment must declare what it's testing and what questions
 * we must ask ourselves after seeing results.
 *
 * NOTE: This file contains TypeScript interfaces (compile-time contracts).
 * Actual reactive state lives in implementations (.svelte.ts files).
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Creature, CreatureState } from '../creatures/types.ts'
import type { World, WorldState } from '../worlds/types.ts'

// ============================================================================
// DATA COLLECTION
// ============================================================================

/**
 * A single data point recorded during an experiment.
 */
export interface DataPoint {
  timestep: number
  creatureState: CreatureState
  worldState: WorldState

  // Computed metrics
  reward: number
  cumulativeReward: number
}

/**
 * Time series data for a metric across the experiment.
 */
export interface TimeSeries {
  name: string
  values: number[]
  timestamps: number[]
}

/**
 * Complete result from running an experiment.
 */
export interface ExperimentResult {
  // Metadata
  experimentName: string
  hypothesis: string
  startTime: Date
  endTime: Date
  totalTimesteps: number

  // Raw data (for detailed analysis)
  data: DataPoint[]

  // Summary metrics
  metrics: Map<string, TimeSeries>

  // Final values
  finalCumulativeReward: number
  finalCreatureState: CreatureState
  finalWorldState: WorldState

  // Reproducibility
  seed: number
  config: Record<string, unknown>
}

// ============================================================================
// ANALYSIS
// ============================================================================

/**
 * Statistical analysis of an experiment result.
 */
export interface Analysis {
  // Basic stats
  meanReward: number
  stdReward: number
  maxReward: number
  minReward: number

  // Trends
  rewardTrend: 'increasing' | 'decreasing' | 'stable' | 'fluctuating'
  learningDetected: boolean

  // Custom metrics (experiment-specific)
  custom: Record<string, number | string | boolean>

  // THE SLAP - must be answered
  slapAnswers: Map<string, string>
}

/**
 * Comparison between two experiment results.
 */
export interface Comparison {
  baseline: ExperimentResult
  treatment: ExperimentResult

  // Statistical comparison
  rewardDifference: number
  percentImprovement: number
  statisticalSignificance: boolean  // Simple threshold test

  // Interpretation
  conclusion: string
}

// ============================================================================
// EXPERIMENT INTERFACE
// ============================================================================

/**
 * An experiment tests a specific hypothesis.
 *
 * THE SLAP questions MUST be answered after every experiment:
 * 1. Did we hard-code this behavior?
 * 2. Did we design the test knowing the answer?
 * 3. Is this actually learning, or just our wiring?
 * 4. Could random weights do this by chance?
 * 5. Would this survive our harshest scrutiny?
 */
export interface Experiment {
  // Identity
  readonly name: string
  readonly description: string

  // Scientific rigor
  readonly hypothesis: string      // What we're testing (falsifiable!)
  readonly expectedOutcome: 'success' | 'failure' | 'unknown'  // Honest prediction
  readonly slapQuestions: string[] // Questions to ask ourselves after

  /**
   * Set up the experiment - create creature and world.
   * Returns the configured environment ready to run.
   */
  setup(): { creature: Creature; world: World }

  /**
   * Run the experiment for a number of timesteps.
   * Returns raw results for analysis.
   *
   * @param timesteps - How many timesteps to run
   * @param onStep - Optional callback for progress/visualization
   */
  run(
    timesteps: number,
    onStep?: (step: number, creature: CreatureState, world: WorldState) => void
  ): Promise<ExperimentResult>

  /**
   * Analyze the results and answer The Slap questions.
   * This is where honesty happens.
   */
  analyze(result: ExperimentResult): Analysis

  /**
   * Clean up resources.
   */
  teardown(): void
}

// ============================================================================
// SUITE INTERFACE
// ============================================================================

/**
 * A suite groups experiments that test one big hypothesis.
 *
 * For example, the "Navigation Suite" tests:
 * "Can creatures learn to navigate better than random/innate?"
 *
 * It does this through multiple experiments:
 * 1. Random baseline (expect: poor navigation)
 * 2. Innate only (expect: basic navigation)
 * 3. Learning enabled (expect: UNKNOWN - real test!)
 */
export interface Suite {
  // Identity
  readonly name: string
  readonly description: string

  // The big question
  readonly hypothesis: string

  // Experiments in this suite (order matters - baselines first)
  readonly experiments: Experiment[]

  /**
   * Run all experiments in sequence.
   * Baselines first, then treatments.
   */
  runAll(
    timestepsPerExperiment: number,
    onExperimentComplete?: (exp: Experiment, result: ExperimentResult) => void
  ): Promise<SuiteResult>

  /**
   * Compare results between experiments.
   * Typically: baseline vs treatment, innate vs learned.
   */
  compare(baseline: ExperimentResult, treatment: ExperimentResult): Comparison
}

/**
 * Complete result from running a suite.
 */
export interface SuiteResult {
  // Metadata
  suiteName: string
  hypothesis: string
  startTime: Date
  endTime: Date

  // Per-experiment results
  results: Map<string, ExperimentResult>  // experimentName → result
  analyses: Map<string, Analysis>         // experimentName → analysis

  // Comparisons
  comparisons: Comparison[]

  // Final verdict
  hypothesisSupported: boolean | 'inconclusive'
  conclusion: string

  // The Slap applied to the whole suite
  suiteSlap: string
}

// ============================================================================
// EXPERIMENT FACTORY
// ============================================================================

export interface ExperimentConfig {
  // Timestep configuration
  dt?: number           // Default: 1.0 ms
  warmupSteps?: number  // Steps before data collection

  // Reproducibility
  seed?: number

  // Data collection
  recordEveryN?: number  // Record data every N steps (default: 1)
}

/**
 * Factory function signature for experiments.
 */
export type ExperimentFactory = (config?: ExperimentConfig) => Experiment

// ============================================================================
// RUNNER CALLBACKS (for visualization)
// ============================================================================

export interface RunnerCallbacks {
  onSuiteStart?: (suite: Suite) => void
  onSuiteEnd?: (suite: Suite, result: SuiteResult) => void

  onExperimentStart?: (experiment: Experiment) => void
  onExperimentEnd?: (experiment: Experiment, result: ExperimentResult) => void

  onStep?: (step: number, creature: CreatureState, world: WorldState) => void
}
