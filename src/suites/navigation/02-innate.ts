/**
 * Experiment 02 - Innate Wiring Only
 *
 * HYPOTHESIS: C. elegans-inspired innate wiring enables basic navigation.
 * EXPECTED OUTCOME: SUCCESS (better than random)
 *
 * This tests whether our biologically-inspired wiring works.
 * The wiring encodes: food stimulus → turn behavior (local search)
 * But it does NOT encode explicit gradient-following.
 *
 * THE SLAP: This should work somewhat. If it fails completely:
 * - Our weight calibration might be wrong
 * - The circuit might not produce the expected behavior
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Experiment, ExperimentResult, Analysis } from '../types.ts'
import { runExperiment, analyzeResult } from '../runner.svelte.ts'
import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createInnateWorm } from './helpers.ts'

// ============================================================================
// EXPERIMENT DEFINITION
// ============================================================================

export function createInnateExperiment(): Experiment {
  let creature: ReturnType<typeof createInnateWorm> | null = null
  let world: ReturnType<typeof createGradient> | null = null

  return {
    name: '02-innate',
    description: 'C. elegans innate wiring without learning',
    hypothesis: 'Innate wiring enables basic navigation via local search behavior',
    expectedOutcome: 'success',
    slapQuestions: [
      'Does the circuit produce the expected behavior (food→turn)?',
      'Is navigation due to local search, not explicit gradient-following?',
      'Are the weights calibrated correctly for chain transmission?',
    ],

    setup() {
      world = createGradient('innate_world', {
        width: 100,
        height: 100,
        sourceX: 75,
        sourceY: 75,
        decayRate: 0.03,
        rewardRadius: 10,
      })

      // Create creature with INNATE wiring, learning DISABLED
      creature = createInnateWorm('innate_creature', {
        scale: 0.5,
        learningEnabled: false,  // No STDP - pure innate
      })

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
      base.custom.expectedToSucceed = true
      base.custom.actuallySucceeded = result.finalCumulativeReward > 0.5

      // Check if behavior matches expectation (food → turns)
      // We'd need to track turn rate near food vs far from food
      // For now, just check overall performance

      base.slapAnswers.set(
        'Does the circuit produce expected behavior?',
        result.finalCumulativeReward > 0.1
          ? 'Yes - some navigation observed'
          : 'No - circuit may not be working as designed'
      )
      base.slapAnswers.set(
        'Is navigation via local search?',
        'To verify: check turn rate correlation with chemical concentration'
      )
      base.slapAnswers.set(
        'Are weights calibrated correctly?',
        base.custom.actuallySucceeded
          ? 'Yes - chain transmission working'
          : 'Maybe not - may need weight adjustment'
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
