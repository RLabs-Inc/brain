/**
 * Experiment 01 - Random Baseline
 *
 * HYPOTHESIS: A creature with RANDOM wiring cannot navigate effectively.
 * EXPECTED OUTCOME: FAILURE (poor navigation)
 *
 * This establishes the baseline. If random wiring can navigate,
 * then our innate wiring means nothing.
 *
 * THE SLAP: This experiment SHOULD fail. If it succeeds, something is wrong:
 * - Either the task is too easy
 * - Or we've accidentally made random wiring good
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Experiment, ExperimentResult, Analysis } from '../types.ts'
import { runExperiment, analyzeResult } from '../runner.svelte.ts'
import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createRandomWorm } from './helpers.ts'

// ============================================================================
// EXPERIMENT DEFINITION
// ============================================================================

export function createRandomExperiment(): Experiment {
  let creature: ReturnType<typeof createRandomWorm> | null = null
  let world: ReturnType<typeof createGradient> | null = null

  return {
    name: '01-random',
    description: 'Baseline with random wiring - should NOT navigate well',
    hypothesis: 'Random wiring cannot navigate a chemical gradient',
    expectedOutcome: 'failure',
    slapQuestions: [
      'If this succeeds, is the task too easy?',
      'Did we accidentally bias random weights toward navigation?',
      'Is the reward signal giving away the answer?',
    ],

    setup() {
      // Create world with gradient
      world = createGradient('random_world', {
        width: 100,
        height: 100,
        sourceX: 75,  // Food in upper right
        sourceY: 75,
        decayRate: 0.03,
        rewardRadius: 10,
      })

      // Create creature with RANDOM wiring (not innate)
      creature = createRandomWorm('random_creature', { scale: 0.5 })

      // Place creature at opposite corner from food
      world.addCreature(creature, { x: 25, y: 25, heading: 0 })

      return { creature, world }
    },

    async run(timesteps: number, onStep?: any): Promise<ExperimentResult> {
      if (!creature || !world) {
        throw new Error('Must call setup() before run()')
      }

      return runExperiment(creature, world, timesteps, { dt: 1.0 }, onStep)
    },

    analyze(result: ExperimentResult): Analysis {
      const base = analyzeResult(result)

      // Add experiment-specific analysis
      base.custom.expectedToFail = true
      base.custom.actuallyFailed = result.finalCumulativeReward < 1.0

      // Answer Slap questions
      base.slapAnswers.set(
        'If this succeeds, is the task too easy?',
        result.finalCumulativeReward > 5.0
          ? 'YES - task may be too easy, random did well'
          : 'No - random performed poorly as expected'
      )
      base.slapAnswers.set(
        'Did we accidentally bias random weights?',
        'No - weights are uniformly random, no bias toward navigation'
      )

      return base
    },

    teardown() {
      creature?.destroy()
      world?.destroy()
      creature = null
      world = null
    },
  }
}
